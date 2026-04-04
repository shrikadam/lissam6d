import os
import time
import torch
import torch.nn as nn
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gc

# --- CONFIGURATION ---
MODEL_NAME = "dinov2_vits14"
IMAGE_SIZE = 518 # Standard DINOv2 resolution (must be multiple of 14)
ONNX_PATH = "../checkpoints/dinov2_s14_fp32.onnx"
FP16_ONNX_PATH = "../checkpoints/dinov2_s14_fp16.onnx"
INT8_ONNX_PATH = "../checkpoints/dinov2_s14_int8.onnx"
TRT_CACHE_DIR = "../checkpoints/trt_cache_dinov2"
DEVICE = 'cuda:0'

def clear_cuda_memory():
    """Forces Python and PyTorch to aggressively release GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class Dinov2Wrapper(nn.Module):
    """Wrapper to handle DINOv2 output cleanly"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model.forward_features(x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

def export_and_quantize_dino():
    print(f"1. Loading DINOv2 ({MODEL_NAME}) from Torch Hub...")
    base_model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)

    # Move to GPU for accurate tracing
    base_model.to(DEVICE)
    base_model.eval
    model = Dinov2Wrapper(base_model)

    # Dummy input must be on the same device
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    output_names = ['cls_token', 'patch_tokens']

    print("\n2. Exporting to ONNX (FP32)...")
    # dynamic_axes removed to ensure perfect TensorRT compilation
    torch.onnx.export(
        model, dummy_input, ONNX_PATH,
        input_names=['image'], output_names=output_names,
        opset_version=18, do_constant_folding=True,
        # dynamic_axes allow you to change image size at runtime
        # dynamic_axes={
        #     'image': {2: 'height', 3: 'width'},
        #     'patch_tokens': {1: 'num_patches'}
        # }
    )
    print(f"  -> FP32 Size: {os.path.getsize(ONNX_PATH) / (1024 * 1024):.2f} MB")

    print("\n3. Exporting to ONNX (FP16 via Autocast)...")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        torch.onnx.export(
            model, dummy_input, FP16_ONNX_PATH,
            input_names=['image'], output_names=output_names,
            opset_version=18, do_constant_folding=True,
            # dynamic_axes allow you to change image size at runtime
            # dynamic_axes={
            #     'image': {2: 'height', 3: 'width'},
            #     'patch_tokens': {1: 'num_patches'}
            # }
        )
    print(f"  -> FP16 Size: {os.path.getsize(FP16_ONNX_PATH) / (1024 * 1024):.2f} MB")

    print("\n4. Quantizing to INT8 (CPU only)...")
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        weight_type=QuantType.QUInt8
    )
    print(f"  -> INT8 Size: {os.path.getsize(INT8_ONNX_PATH) / (1024 * 1024):.2f} MB")

def test_inference_dino(image_path):
    print(f"\n--- Benchmarking DINOv2 ONNX on {image_path} ---")
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)

    # 1. Preprocess image
    original_img = cv2.imread(image_path)
    if original_img is None: raise ValueError('Image not found')
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Force a 3-channel RGB
    if len(original_img.shape) == 3 and original_img.shape[2] == 4:
        # It's an RGBA image (4 channels), drop the alpha
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2RGB)
    elif len(original_img.shape) == 3 and original_img.shape[2] == 3:
        # It's a standard BGR image (3 channels)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        # It's a grayscale image (2 dimensions)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    # A. Resize using OpenCV (Much faster than PIL)
    img_resized = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    # B. Scale to [0, 1] range
    img_float = img_resized.astype(np.float32) / 255.0

    # C. Apply ImageNet Normalization via NumPy broadcasting
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std

    # D. Transpose from (H, W, C) to (1, C, H, W) for ONNX
    x_fp32 = img_normalized.transpose(2, 0, 1)[None, ...]

    # 2. Benchmark Configurations
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
    sess_options.log_severity_level = 3

    # 3. Benchmark Loop
    for cfg in configs:
        print(f"\n[{cfg['name']}] Initializing Session...")
        if not os.path.exists(cfg['path']):
            print(f"   -> Skipping! File {cfg['path']} not found.")

        ort_sess = onnxruntime.InferenceSession(
            cfg['path'],
            sess_options=sess_options,
            providers=cfg['provider']
        )

        input_dict = {'image': cfg['input']}

        # --- WARMUP ---
        for _ in range(5):
            ort_sess.run(None, input_dict)

        # --- BENCHMARK ---
        num_runs = 20
        start_time = time.perf_counter()
        for _ in range(num_runs):
            outputs = ort_sess.run(None, input_dict)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / num_runs) * 1000
        fps = 1000 / avg_time_ms
        print(f"  -> Avg Time {avg_time_ms:.2f} ms ({fps:.2f}) FPS")

        # --- VISUALIZE PCA ---
        patch_tokens = outputs[1]
        features = patch_tokens[0] # Shape (1368, 384)

        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features)
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        pca_features = (pca_features * 255).astype(np.uint8)

        grid_size = IMAGE_SIZE // 14
        vis_image = pca_features.reshape(grid_size, grid_size, 3)
        vis_image_resized = cv2.resize(vis_image, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original_img)
        ax[0].set_title("Original")
        ax[0].axis('off')

        ax[1].imshow(vis_image_resized)
        ax[1].set_title(f"PCA | {cfg['name']} | {avg_time_ms:.1f}ms")
        ax[1].axis('off')

        out_name = f"vis_dino_{cfg['name']}.jpg"
        plt.savefig(out_name, bbox_inches='tight')
        plt.close()

        # --- CLEANUP ---
        del ort_sess, outputs, pca_features
        clear_cuda_memory()

if __name__ == "__main__":
    # export_and_quantize_dino()
    test_inference_dino("traffic.png")