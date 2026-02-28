import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import cv2
import time
import torch
import numpy as np
import argparse

# SAM 2 Native Imports
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import gc
def clear_cuda_memory():
    """Forces Python and PyTorch to aggressively release GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

from collections import OrderedDict
import torchvision.transforms as T

class SAM2CameraStreamWrapper:
    def __init__(self, predictor, max_memory_frames=60):
        """
        Wraps the SAM 2 Video Predictor to accept live OpenCV frames.
        max_memory_frames: How many past frames to keep in VRAM. SAM 2 usually 
        only looks back ~7 frames, so 60 is plenty and prevents OOM crashes.
        """
        self.predictor = predictor
        self.max_memory_frames = max_memory_frames
        self.inference_state = None
        self.frame_idx = 0

    @torch.inference_mode()
    def init_stream(self, first_frame_rgb, golden_mask):
        """Called once when transitioning from ACQUISITION to TRACKING"""
        # 1. Manually construct the SAM 2 inference state
        self.inference_state = {
            "images": {}, # Dictionary instead of list for garbage collection
            "num_frames": 1,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": first_frame_rgb.shape[0],
            "video_width": first_frame_rgb.shape[1],
            "device": self.predictor.device,
            "storage_device": self.predictor.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {}
        }
        self.frame_idx = 0
        
        # 2. Process and append the first frame
        self._append_frame(first_frame_rgb)
        
        # 3. Inject the golden mask from DINOv2
        # SAM 2 API uses add_new_mask which requires a tensor/numpy array of shape (H, W)
        with torch.autocast(device_type=self.predictor.device.type, dtype=torch.bfloat16):
            _, _, video_res_masks = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=self.frame_idx,
                obj_id=1,
                mask=golden_mask
            )
        return (video_res_masks[0, 0] > 0.0).cpu().numpy()

    @torch.inference_mode()
    def track_next_frame(self, frame_rgb):
        """Called on every subsequent frame to track the object."""
        self.frame_idx += 1
        self.inference_state["num_frames"] = self.frame_idx + 1
        
        # 1. Process and append the new live frame
        self._append_frame(frame_rgb)
        
        # 2. Preflight (Consolidates memory bank)
        with torch.autocast(device_type=self.predictor.device.type, dtype=torch.bfloat16):
            self.predictor.propagate_in_video_preflight(self.inference_state)
            
            # 3. Manually run the single frame inference
            batch_size = len(self.inference_state["obj_ids"])
            current_out, pred_masks_gpu = self.predictor._run_single_frame_inference(
                inference_state=self.inference_state,
                output_dict=self.inference_state["output_dict"],
                frame_idx=self.frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True
            )
        
        # 4. Save outputs to the state dictionary
        self.inference_state["output_dict"]["non_cond_frame_outputs"][self.frame_idx] = current_out
        self.predictor._add_output_per_object(self.inference_state, self.frame_idx, current_out, "non_cond_frame_outputs")
        self.inference_state["frames_already_tracked"][self.frame_idx] = {"reverse": False}
        
        # 5. Resize to original video resolution
        _, video_res_masks = self.predictor._get_orig_video_res_output(self.inference_state, pred_masks_gpu)
        
        # 6. Garbage collection
        self._clear_old_memory()
        
        tracked_mask = video_res_masks[0, 0] > 0.0
        score = current_out["object_score_logits"][0, 0].item()
        
        return tracked_mask.cpu().numpy(), score

    def _append_frame(self, frame_rgb):
        """Replicates SAM 2's internal load_video_frames preprocessing."""
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = T.Resize((self.predictor.image_size, self.predictor.image_size), antialias=True)(img_tensor)
        
        # SAM 2 Standard Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        self.inference_state["images"][self.frame_idx] = img_tensor.to(self.inference_state["storage_device"])

    def _clear_old_memory(self):
        """Deletes old frames to prevent VRAM explosion."""
        old_frame_idx = self.frame_idx - self.max_memory_frames
        if old_frame_idx in self.inference_state["images"]:
            del self.inference_state["images"][old_frame_idx]
            
            if old_frame_idx in self.inference_state["cached_features"]:
                del self.inference_state["cached_features"][old_frame_idx]

            if old_frame_idx in self.inference_state["output_dict"]["non_cond_frame_outputs"]:
                del self.inference_state["output_dict"]["non_cond_frame_outputs"][old_frame_idx]
                
            for obj_idx in self.inference_state["output_dict_per_obj"]:
                if old_frame_idx in self.inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]:
                    del self.inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"][old_frame_idx]

class LeanDINOv2Descriptor:
    def __init__(self, device="cuda"):
        self.device = device
        # Load the raw DINOv2 model directly from Hub
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.model.eval()
        
        # Standard ImageNet normalization for DINOv2
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.target_size = 224 # DINOv2 expected crop size

    @torch.inference_mode()
    def process_and_get_descriptors(self, image_rgb_tensor, masks, boxes):
        """
        Takes the full image and SAM2 bounding boxes/masks. 
        Crops, masks, resizes, and feeds them to DINOv2 in memory-safe chunks.
        """
        N = len(masks)
        if N == 0: return None, None
        
        # CRITICAL FIX: Do NOT normalize the image yet. Keep it in [0, 1] range.
        
        processed_crops = []
        for i in range(N):
            x, y, w, h = boxes[i].int().tolist()
            if w <= 0 or h <= 0: continue
            
            crop_img = image_rgb_tensor[:, y:y+h, x:x+w]
            crop_mask = masks[i:i+1, y:y+h, x:x+w]
            
            # Apply mask to image (Background becomes TRUE [0,0,0] black)
            masked_crop = crop_img * crop_mask
            
            sq_size = max(w, h)
            pad_w = sq_size - w
            pad_h = sq_size - h
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            
            padded_crop = F.pad(masked_crop, padding, value=0)
            resized_crop = F.interpolate(padded_crop.unsqueeze(0), size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            
            processed_crops.append(resized_crop.squeeze(0))

        if not processed_crops: return None, None
        
        batch_crops = torch.stack(processed_crops) # [N, 3, 224, 224]

        # --- DEBUG: VISUALIZE DINOV2 INPUTS (Now perfectly black!) ---
        # import torchvision
        # debug_crops = batch_crops[:16].float().cpu() 
        # torchvision.utils.save_image(debug_crops, "debug_dinov2_inputs.jpg", nrow=4)
        # -------------------------------------------------------------

        # NOW apply ImageNet normalization to the batch
        batch_crops = self.normalize(batch_crops)
        
        # CRITICAL FIX 2: Chunk the DINOv2 forward pass to prevent OOM
        chunk_size = 16
        all_cls = []
        all_patch = []
        
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk = batch_crops[start_idx:end_idx].to(torch.bfloat16) # Ensure bfloat16
            
            features = self.model.forward_features(chunk)
            all_cls.append(features["x_norm_clstoken"])
            all_patch.append(features["x_norm_patchtokens"])
            
        # Stitch chunks back together
        cls_tokens = torch.cat(all_cls, dim=0)
        patch_tokens = torch.cat(all_patch, dim=0)
        
        return cls_tokens, patch_tokens

    @torch.inference_mode()
    def compute_semantic_match(self, query_cls_tokens, template_cls_tokens):
        """
        Performs Cosine Similarity matching between live proposals and pre-computed templates.
        """
        # query: [N, D], template: [T, D]
        query_norm = F.normalize(query_cls_tokens, dim=-1)
        template_norm = F.normalize(template_cls_tokens, dim=-1)
        
        # Calculate similarity matrix [N, T]
        similarity_matrix = torch.matmul(query_norm, template_norm.T)
        
        # Get the best template score for each proposal
        best_scores, best_template_idx = similarity_matrix.max(dim=1)
        return best_scores, best_template_idx
        
class RealTimeSegmentor:
    def __init__(self, seg_generator, img_descriptor, seg_tracker, device="cuda"):
        self.device = device
        self.seg_generator = seg_generator
        self.img_descriptor = img_descriptor
        self.seg_tracker = SAM2CameraStreamWrapper(seg_tracker)
        
        self.state = "ACQUISITION"
        self.templates_cls = None
        self.tracking_threshold = 0.0 # SAM2 logit tracking threshold
        self.acquisition_threshold = 0.5 # DINOv2 Cosine Similarity threshold

    def load_templates(self, template_dir):
        """
        Loads rgb and mask pairs from template_dir, crops them using the mask bounding box,
        pads to a square, resizes to 224x224, and computes their DINOv2 CLS tokens.
        """
        import glob
        from PIL import Image
        import torch.nn.functional as F

        print(f"Loading templates from {template_dir}...")
        
        # Find all rgb template files
        num_templates = len(glob.glob(os.path.join(template_dir, 'rgb_*.png')))
        if num_templates == 0:
            raise FileNotFoundError(f"No rgb_*.png templates found in {template_dir}")
            
        processed_crops = []
        
        for idx in range(num_templates):
            rgb_path = os.path.join(template_dir, f'rgb_{idx}.png')
            mask_path = os.path.join(template_dir, f'mask_{idx}.png')
            
            image = Image.open(rgb_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            # Get bounding box (left, upper, right, lower)
            bbox = mask.getbbox()
            if bbox is None:
                continue # Skip empty masks
                
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # Convert to tensors (C, H, W) in [0, 1]
            img_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(np.array(mask)).float() / 255.0
            
            # Apply mask (black out the background)
            masked_img = img_t * mask_t
            
            # Crop to the bounding box
            crop_img = masked_img[:, y1:y2, x1:x2]
            
            # Pad to a perfect square to preserve aspect ratio
            sq_size = max(w, h)
            pad_w = sq_size - w
            pad_h = sq_size - h
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2) 
            
            padded_crop = F.pad(crop_img, padding, value=0)
            
            # Resize to DINOv2's expected 224x224
            resized_crop = F.interpolate(
                padded_crop.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            processed_crops.append(resized_crop.squeeze(0))
            
        if not processed_crops:
            raise ValueError("No valid templates could be processed (masks might be empty).")
            
        # Stack into a single batch tensor: [T, 3, 224, 224]
        template_tensor_batch = torch.stack(processed_crops).to(self.device)
        
        # Apply DINOv2 ImageNet Normalization
        template_tensor_batch = self.img_descriptor.normalize(template_tensor_batch)
        
        # Pass the batch through DINOv2 to get the reference CLS tokens
        with torch.inference_mode(), torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # Ensure the tensor matches the autocast type
            template_tensor_batch = template_tensor_batch.to(torch.bfloat16)

            features = self.img_descriptor.model.forward_features(template_tensor_batch)
            self.templates_cls = features["x_norm_clstoken"]
            
        print(f"Successfully loaded and computed descriptors for {self.templates_cls.shape[0]} templates.")
    
    def process_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        start_time = time.perf_counter()

        if self.state == "ACQUISITION":
            mask = self._run_acquisition(frame_rgb)
            if mask is not None:
                self.state = "TRACKING"
                # Pass the first frame and the golden mask to initialize SAM 2's memory bank
                result_mask = self.seg_tracker.init_stream(frame_rgb, mask)
                print("Target ACQUIRED. Switching to TRACKING mode.")
            else:
                result_mask = None
                
        elif self.state == "TRACKING":
            # Track using the sliding-window memory
            mask, logit_score = self.seg_tracker.track_next_frame(frame_rgb)
            
            # State Transition: If confidence drops below threshold or mask vanishes
            if logit_score < self.tracking_threshold or mask.sum() < 50:
                print(f"Track Lost! Score: {logit_score:.2f}. Reverting to ACQUISITION.")
                self.state = "ACQUISITION"
                result_mask = None
            else:
                result_mask = mask

        fps = 1.0 / (time.perf_counter() - start_time)
        return result_mask, self.state, fps

    def _run_acquisition(self, frame_rgb):
        """Pure, unadulterated SAM AMG + DINOv2 Matching"""
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # 1. Generate Masks with SAM 2 AMG
            # Output is a list of dicts: [{'segmentation': mask, 'bbox': [x,y,w,h], ...}]
            sam_outputs = self.seg_generator.generate(frame_rgb)

            if not sam_outputs: return None

            # Format for DINOv2
            masks = torch.tensor(np.array([out['segmentation'] for out in sam_outputs]), device=self.device, dtype=torch.float32)
            boxes = torch.tensor(np.array([out['bbox'] for out in sam_outputs]), device=self.device, dtype=torch.float32)
            
            # Prepare image tensor (C, H, W) in [0, 1] range
            image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(self.device) / 255.0

            image_tensor = image_tensor.to(torch.bfloat16)
            
            # 2. Extract Descriptors
            query_cls, _ = self.img_descriptor.process_and_get_descriptors(image_tensor, masks, boxes)
            if query_cls is None: return None

            # 3. Match against Templates
            best_scores, best_template_idx = self.img_descriptor.compute_semantic_match(query_cls, self.templates_cls)
            # --- DEBUG: NUMERICAL SCORE SPREAD ---
            # # Sort scores to see the top 5 highest matches
            # sorted_scores, sorted_indices = torch.sort(best_scores, descending=True)
            # top_5_scores = sorted_scores[:5].tolist()
            
            # print(f"DEBUG: Top 5 Cosine Similarities: {[round(s, 3) for s in top_5_scores]}")
            # print(f"DEBUG: Mean Score: {best_scores.mean().item():.3f} | Min: {best_scores.min().item():.3f}")
            # -------------------------------------
            
            # 4. Find the winner
            winner_idx = torch.argmax(best_scores).item()
            if best_scores[winner_idx] > self.acquisition_threshold:
                print(f"Target Found! DINOv2 Score: {best_scores[winner_idx]:.2f}")
                # --- DEBUG: VISUALIZE THE WINNING MATCH ---
                # win_mask = sam_outputs[winner_idx]['segmentation']
                # win_bbox = sam_outputs[winner_idx]['bbox'] # [x, y, w, h]
                # x, y, w, h = [int(v) for v in win_bbox]
                
                # # Crop the live frame
                # live_crop = frame_rgb[y:y+h, x:x+w].copy()
                # live_crop[~win_mask[y:y+h, x:x+w]] = 0 # Apply mask
                # live_crop = cv2.resize(live_crop, (224, 224))
                
                # # We matched against self.templates_cls[best_template_idx[winner_idx]]
                # # (Assuming you saved the template images during load_templates)
                # matched_template_idx = best_template_idx[winner_idx].item()
                # score = best_scores[winner_idx].item()
                
                # cv2.imwrite(f"debug_match_score_{score:.2f}.jpg", cv2.cvtColor(live_crop, cv2.COLOR_RGB2BGR))
                # print(f"DEBUG: Saved winning live crop to 'debug_match_score_{score:.2f}.jpg'")
                # print(f"DEBUG: This matched against Template Index #{matched_template_idx}")
                # ------------------------------------------
                return sam_outputs[winner_idx]['segmentation'] # Return boolean numpy mask
            
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time SAM6D Tracking")
    parser.add_argument("--template_dir", required=True, help="Path to object templates")
    parser.add_argument("--video_source", default="0", help="Camera ID (e.g., 0) or path to video file")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 1. Initialize Models
    # ==========================================
    # We use SAM 2 Tiny for maximum FPS on the Jetson
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
    sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    img_descriptor = LeanDINOv2Descriptor(device=DEVICE)
    img_descriptor.model = img_descriptor.model.to(torch.bfloat16)
    # Build the SAM 2 Video Predictor for the Tracking phase
    seg_tracker = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=DEVICE).to(torch.bfloat16)
    # The AMG will ignore the video memory modules and just use the base image methods.
    seg_generator = SAM2AutomaticMaskGenerator(
        model=seg_tracker,
        points_per_side=16,     # Drops total points from 1024 to just 256!
        points_per_batch=16,    # Processes only 16 points at a time in VRAM
        stability_score_thresh=0.92, # Slightly relaxed to ensure we catch the object
        min_mask_region_area=100     # Ignore tiny specs of dust/noise
    )

    # Initialize our State Machine Tracker
    pipeline = RealTimeSegmentor(seg_generator, img_descriptor, seg_tracker, device=DEVICE)

    # ==========================================
    # 2. Offline Preparation
    # ==========================================
    pipeline.load_templates(args.template_dir)

    # ==========================================
    # 3. Real-Time Inference Loop
    # ==========================================
    # Handle webcam (int) or video file (str)
    source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    cap = cv2.VideoCapture(source)
    # Grab the width and height from the camera stream
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # If reading from a video file, use: fps_out = cap.get(cv2.CAP_PROP_FPS)
    fps_out = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('tracking_output.mp4', fourcc, fps_out, (frame_width, frame_height))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    clear_cuda_memory()
    print("Starting Real-Time Tracking Loop. Press 'ESC' to exit.")
    
    num_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("End of stream.")
                break

            mask, state, fps = pipeline.process_frame(frame)
            num_frames += 1
            # Vis
            if mask is not None:
                # Bulletproof check: if it's a tensor, move it to CPU and make it numpy
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                
                # Ensure it's boolean
                bool_mask = mask > 0 
                
                # Apply the blue overlay
                frame[bool_mask] = frame[bool_mask] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5
                
            cv2.putText(frame, f"State: {state} | FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            # Write the frame to the video file instead of showing it
            out_video.write(frame)
            
            # Print to terminal so you know it hasn't frozen
            print(f"Processing Frame {num_frames} | State: {state} | FPS: {fps:.1f}", end='\r')

    except KeyboardInterrupt:
        print("\nManually stopped recording via Ctrl+C.")

    cap.release()
    out_video.release() # CRITICAL: This finalizes and saves the mp4 file
    print("Video saved successfully to tracking_output.mp4")