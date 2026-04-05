import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

# SAM 2 Imports
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- CONFIGURATION ---
DEVICE = "cuda"
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
OUTPUT_DIR = "masks"

def show_anns(anns):
    """Helper function to overlay SAM's masks on a Matplotlib plot."""
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.45]]) 
        img[m] = color_mask
    ax.imshow(img)

def save_individual_masks(anns, output_folder):
    """Saves each mask as an individual black and white image."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    print(f"Saving {len(anns)} individual masks...")
    for i, ann in enumerate(anns):
        # Convert boolean mask to uint8 (0 and 255)
        mask_data = ann['segmentation'].astype(np.uint8) * 255
        file_path = os.path.join(output_folder, f"mask_{i:04d}.png")
        cv2.imwrite(file_path, mask_data)

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="SAM 2 Automatic Mask Generation")
    parser.add_argument("--image", type=str, default="../media/image_rgb3.png", help="Path to input image")
    args = parser.parse_args()

    print("1. Loading SAM 2 Video Predictor (in float16)...")
    seg_tracker = build_sam2_video_predictor(
        SAM2_CONFIG, 
        SAM2_CHECKPOINT, 
        device=DEVICE
    ).to(torch.float16) 

    print("2. Initializing Automatic Mask Generator...")
    amg = SAM2AutomaticMaskGenerator(
        model=seg_tracker,
        points_per_batch=32, 
        stability_score_thresh=0.95 
    )

    print(f"3. Loading image: {args.image}")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load {args.image}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("4. Running AMG Inference...")
    start_time = time.perf_counter()
    
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        masks = amg.generate(image_rgb)
        
    end_time = time.perf_counter()
    print(f" -> AMG finished in {end_time - start_time:.2f} seconds.")
    print(f" -> Found {len(masks)} masks.")

    # --- NEW CAPABILITY: SAVE MASKS ---
    save_individual_masks(masks, OUTPUT_DIR)

    print("5. Visualizing outputs...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    show_anns(masks)
    plt.axis('off')
    
    out_path = "amg_debug_output.jpg"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved summary visualization to {out_path}")

if __name__ == "__main__":
    main()