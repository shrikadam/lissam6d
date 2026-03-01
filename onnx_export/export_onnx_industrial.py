import os
import torch
import torch.nn as nn
import math
import gc

# SAM 2 Native Imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
DEVICE = 'cuda:0'
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml" 
EXPORT_DIR = "../checkpoints/industrial_onnx"

os.makedirs(EXPORT_DIR, exist_ok=True)

def clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# 1. DINOv2 WRAPPER & EXPORT
# ==========================================
class Dinov2IndustrialWrapper(nn.Module):
    """Outputs ONLY the CLS token. Patch tokens are dropped to save C++ memory bus bandwidth."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model.forward_features(x)
        return out["x_norm_clstoken"]

def export_dinov2_fp16():
    print("\n--- Exporting DINOv2 (FP16, Dynamic Batch) ---")
    base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()
    model = Dinov2IndustrialWrapper(base_model)

    # Dummy input: Batch size of 2 to trace the dynamic axis properly
    dummy_input = torch.randn(2, 3, 224, 224, device=DEVICE)
    onnx_path = os.path.join(EXPORT_DIR, "dinov2_vits14_fp16.onnx")

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['image_crops'], 
            output_names=['cls_tokens'],
            opset_version=18, 
            do_constant_folding=True,
            # CRITICAL: Allow the batch size (dim 0) to change at C++ runtime
            dynamic_axes={
                'image_crops': {0: 'batch_size'},
                'cls_tokens': {0: 'batch_size'}
            }
        )
    print(f"✅ Saved: {onnx_path} ({os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB)")
    clear_cuda_memory()

# ==========================================
# 2. SAM 2 ENCODER WRAPPER & EXPORT
# ==========================================
class SAM2EncoderIndustrialWrapper(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.model = predictor.model

    def forward(self, x):
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        
        if self.model.no_mem_embed is not None:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            
        feats = []
        for feat in vision_feats:
            hw_flat = feat.shape[0]
            h = w = math.isqrt(hw_flat)
            formatted_feat = feat.permute(1, 2, 0).view(x.shape[0], -1, h, w)
            feats.append(formatted_feat)
            
        return feats[2], feats[0], feats[1]

def export_sam2_encoder_fp16():
    print("\n--- Exporting SAM 2 Image Encoder (FP16, Static Batch) ---")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    model = SAM2EncoderIndustrialWrapper(predictor).eval()

    # Dummy input: Strictly 1x3x1024x1024 for predictable TensorRT memory pre-allocation
    dummy_input = torch.randn(1, 3, 1024, 1024, device=DEVICE)
    onnx_path = os.path.join(EXPORT_DIR, "sam2_encoder_fp16.onnx")

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['image'], 
            output_names=['image_embed', 'high_res_0', 'high_res_1'],
            opset_version=18, 
            do_constant_folding=True
            # No dynamic axes! We want TensorRT to heavily optimize this static shape.
        )
    print(f"✅ Saved: {onnx_path} ({os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB)")
    clear_cuda_memory()

if __name__ == "__main__":
    print("Initializing Industrial Export Pipeline...")
    export_dinov2_fp16()
    export_sam2_encoder_fp16()
    print("\n🎉 Phase 1 Complete! Models are ready for C++.")