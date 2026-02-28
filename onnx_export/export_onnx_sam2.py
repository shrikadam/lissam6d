import os
import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2, build_sam2_object_tracker
from sam2.sam2_image_predictor import SAM2ImagePredictor
import math
import time
import gc

def clear_cuda_memory():
    """Forces Python and PyTorch to aggressively release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# --- CONFIGURATION ---
# Adjust these paths to where you saved your SAM 2 files
CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "./configs/sam2.1/sam2.1_hiera_t.yaml" 
ONNX_PATH = "../checkpoints/sam2.1_tiny_encoder.onnx"
FP16_ONNX_PATH = "../checkpoints/sam2.1_tiny_encoder_fp16.onnx"
INT8_ONNX_PATH = "../checkpoints/sam2.1_tiny_encoder_int8.onnx"
TRT_CACHE_DIR = "../checkpoints/trt_cache_sam2"
DEVICE = 'cuda:0'
   
class SAM2EncoderWrapper(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.model = predictor.model

    def forward(self, x):
        # 1. Run backbone (Outputs 1 feature for Tiny)
        backbone_out = self.model.forward_image(x)
        
        # 2. Run Neck (FPN) (Always outputs 3 features: 256x256, 128x128, 64x64)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        
        # 3. Add 'no memory' embedding to the lowest-res feature
        if self.model.no_mem_embed is not None:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            
        # 4. Reshape flattened tensors back to (B, C, H, W)
        feats = []
        for feat in vision_feats:
            # feat shape is (H*W, B, C)
            hw_flat = feat.shape[0]
            
            # Since H == W, we just take the square root of H*W
            # math.isqrt safely returns an integer for the ONNX tracer
            h = w = math.isqrt(hw_flat)
            
            # Reshape from (H*W, B, C) -> (B, C, H, W)
            formatted_feat = feat.permute(1, 2, 0).view(x.shape[0], -1, h, w)
            feats.append(formatted_feat)
            
        # Return them in the exact order the decoder expects:
        # feats[2] = 64x64 (image_embed)
        # feats[0] = 256x256 (high_res_0)
        # feats[1] = 128x128 (high_res_1)
        return feats[2], feats[0], feats[1]

def export_and_quantize_sam2():
    print("1. Loading SAM 2...")
    sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE)
    
    # We instantiate the Predictor so we can access its specific logic
    predictor = SAM2ImagePredictor(sam2_model)
    # Pass the predictor to our Bulletproof Wrapper
    model = SAM2EncoderWrapper(predictor)
    model.eval()

    print("\n2. Exporting to ONNX (FP32)...")
    dummy_input = torch.randn(1, 3, 1024, 1024).to(DEVICE)
    output_names = ['image_embed', 'high_res_0', 'high_res_1']
    torch.onnx.export(
        model, dummy_input, ONNX_PATH,
        input_names=['image'], output_names=output_names,
        opset_version=18, do_constant_folding=True
    )
    size_fp32 = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"   -> FP32 Size: {size_fp32:.2f} MB")
    
    print("\n3. Exporting to ONNX (FP16 via Autocast)...")
    # Ensure model and dummy input start as FP32 on CUDA
    model = model.float() 
    dummy_input = torch.randn(1, 3, 1024, 1024).to(DEVICE)
    # Use PyTorch's native mixed-precision context
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        torch.onnx.export(
            model, dummy_input, FP16_ONNX_PATH,
            input_names=['image'], output_names=output_names,
            opset_version=18, do_constant_folding=True
        )
    size_fp16 = os.path.getsize(FP16_ONNX_PATH) / (1024 * 1024)
    print(f"   -> FP16 Size: {size_fp16:.2f} MB")
    
    print("\n4. Quantizing to INT8...")
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        weight_type=QuantType.QUInt8
    )
    print("Export Complete!")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def test_inference_sam2(image_path):
    print(f"\n--- Benchmarking SAM 2 ONNX Encoders on {image_path} ---")

    # 1. Load Original PyTorch Model (For the Decoder)
    sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)

    # 2. Preprocess Image
    image = cv2.imread(image_path)
    if image is None: raise ValueError("Image not found")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    input_size = 1024
    img_resized = cv2.resize(image, (input_size, input_size))
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    
    # Create the base FP32 input
    x_fp32 = (img_resized - mean) / std
    x_fp32 = x_fp32.transpose(2, 0, 1).astype(np.float32)[None, :, :, :]

    # 3. Define the benchmark configurations
    configs = [
        {
            "name": "FP32_CUDA",
            "path": ONNX_PATH,
            "provider": ['CUDAExecutionProvider'],
            "input": x_fp32
        },
        {
            "name": "FP16_CUDA",
            "path": FP16_ONNX_PATH,
            "provider": ['CUDAExecutionProvider'],
            "input": x_fp32
        },
        {
            "name": "FP32_TRT",
            "path": ONNX_PATH,
            "provider": [
                ('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': TRT_CACHE_DIR,
                    'trt_fp16_enable': False
                }),
                'CUDAExecutionProvider'],
            "input": x_fp32
        },
        {
            "name": "FP16_TRT",
            "path": FP16_ONNX_PATH,
            "provider": [
                ('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': TRT_CACHE_DIR,
                    'trt_fp16_enable': False # Model is already FP16 native due to AMP, no need to optimize
                }),
                'CUDAExecutionProvider'],
            "input": x_fp32
        },
        {
            "name": "INT8_CPU",
            "path": INT8_ONNX_PATH,
            "provider": ['CPUExecutionProvider'],
            "input": x_fp32
        }
    ]

    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = 3  # Mute warnings
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Setup the predictor transforms once
    from sam2.utils.transforms import SAM2Transforms
    predictor._transforms = SAM2Transforms(
        resolution=input_size, mask_threshold=0.0, 
        max_hole_area=0.0, max_sprinkle_area=0.0
    )
    
    h, w = image.shape[:2]
    input_point = np.array([[w // 3, 3 * h // 4]])
    input_label = np.array([1]) # Foreground click

    # 4. Run the Benchmark Loop
    for cfg in configs:
        print(f"\n[{cfg['name']}] Initializing Session...")
        if not os.path.exists(cfg['path']):
            print(f"  -> Skipping! File {cfg['path']} not found.")
            continue
            
        ort_sess = onnxruntime.InferenceSession(
            cfg['path'], 
            sess_options=sess_options,
            providers=cfg['provider']
        )
        
        input_dict = {"image": cfg['input']}

        # --- WARMUP ---
        # Crucial for CUDA to finish kernel compilation and memory allocation
        for _ in range(5):
            ort_sess.run(None, input_dict)

        # --- BENCHMARK ---
        num_runs = 10
        start_time = time.perf_counter()
        for _ in range(num_runs):
            numpy_features = ort_sess.run(None, input_dict)
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / num_runs) * 1000
        fps = 1000 / avg_time_ms
        print(f"  -> Encoder Avg Time: {avg_time_ms:.2f} ms ({fps:.1f} FPS)")

        # --- DECODER INFERENCE ---
        # Convert outputs to torch tensors. We explicitly cast to float32 here 
        # so the PyTorch decoder handles the FP16 ONNX outputs safely before autocast takes over.
        # features = [torch.from_numpy(f).to(device=DEVICE, dtype=torch.float32) for f in numpy_features]
        
        with torch.inference_mode():
            # Force contiguous memory to remove any CUDA EP padding/stride artifacts
            features = [
                torch.from_numpy(np.ascontiguousarray(f)).to(device=DEVICE, dtype=torch.float32) 
                for f in numpy_features
            ]

            features_dict = {
                "image_embed": features[0],
                "high_res_feats": [features[1], features[2]]
            }
            
            predictor._features = features_dict
            predictor._is_image_set = True
            predictor._orig_hw = [(image.shape[0], image.shape[1])]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )

        # --- VISUALIZE AND SAVE ---
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        plt.plot(input_point[:, 0], input_point[:, 1], 'go')
        plt.axis('off')
        
        out_name = f"vis_sam2_{cfg['name']}.jpg"
        plt.title(f"{cfg['name']} | Score: {scores[0]:.2f} | Enc Time: {avg_time_ms:.1f}ms")
        plt.savefig(out_name, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved visualization to {out_name}")

        print(f"[{cfg['name']}] Cleaning up memory...")
        
        # Delete heavy variables explicitly
        del ort_sess
        del numpy_features
        del features
        del features_dict
        del masks, scores, logits
        
        # Call our new cleanup function
        clear_cuda_memory()

if __name__ == "__main__":
    # export_and_quantize_sam2()
    clear_cuda_memory()
    test_inference_sam2("giraffe.jpg")