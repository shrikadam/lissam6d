import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# SAM 2 Imports
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- CONFIGURATION ---
IMAGE_PATH = "servo.jpg" # Point this to a frame from your video!
DEVICE = "cuda"
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

def show_anns(anns):
    """Helper function to overlay SAM's masks on a Matplotlib plot."""
    if len(anns) == 0:
        return
    
    # Sort annotations by area to draw largest masks first (background), then smaller ones
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.45]]) # Random color, 45% opacity
        img[m] = color_mask
    ax.imshow(img)

def main():
    print("1. Loading SAM 2 Video Predictor (in bfloat16)...")
    # Cast the model immediately upon loading
    # BFloat16 prevents NaN overflows in Attention layers better than Float16
    seg_tracker = build_sam2_video_predictor(
        SAM2_CONFIG, 
        SAM2_CHECKPOINT, 
        device=DEVICE
    ).to(torch.bfloat16) 

    print("2. Initializing Automatic Mask Generator...")
    # Adjust points_per_batch to save VRAM (default is 64). 
    # Lowering this uses less memory but takes slightly longer.
    amg = SAM2AutomaticMaskGenerator(
        model=seg_tracker,
        points_per_batch=32, 
        stability_score_thresh=0.95 # Higher threshold = cleaner, fewer junk masks
    )

    print(f"3. Loading image: {IMAGE_PATH}")
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load {IMAGE_PATH}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("4. Running AMG Inference...")
    start_time = time.perf_counter()
    
    # Force the forward pass to use bfloat16 autocast
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks = amg.generate(image_rgb)
        
    end_time = time.perf_counter()
    print(f" -> AMG finished in {end_time - start_time:.2f} seconds.")
    print(f" -> Found {len(masks)} masks.")

    print("5. Visualizing outputs...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    show_anns(masks)
    plt.axis('off')
    
    # Save the visualization
    out_path = "amg_debug_output.jpg"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()