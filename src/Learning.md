# Learning.md: Zero-Shot Sim2Real Tracking on Jetson Edge

## 1. The Core Concept: "Detect-Once, Track-Continuously"
Running heavy semantic segmentation on every frame is impossible on edge devices. Instead, we architected a **Finite State Machine** with two phases:
* **ACQUISITION (Heavy):** Uses SAM 2's Automatic Mask Generator (AMG) to segment the entire scene, extracts DINOv2 embeddings for every mask, and mathematically compares them (Cosine Similarity) against offline CAD templates to find the target.
* **TRACKING (Light):** Once the target is acquired, the "golden mask" is handed off to SAM 2's Video Predictor. SAM 2 uses its internal spatial memory bank to propagate that mask across subsequent frames at high speed, bypassing DINOv2 entirely.

---

## 2. Environment & Dependency Hurdles

### Hurdle: The `xformers` Crash (Environment Mismatch)
* **The Problem:** Jetson `pip` installed an `xformers` wheel compiled for Python 3.12, but the environment was 3.10. When DINOv2 tried to use memory-efficient attention, it failed to load the C++ backend and crashed.
* **The Fix:** Nuked `xformers` entirely. PyTorch 2.0+ has native `F.scaled_dot_product_attention` (SDPA) which is highly optimized. DINOv2 and SAM 2 automatically fall back to it seamlessly.

### Hurdle: Python "Shadowing" & Missing `_C` Binaries
* **The Problem:** Running a script named `realtime_inference.py` inside a directory that also contains a folder named `sam2/` causes Python to import the local folder instead of the installed library. Furthermore, a warning popped up about a missing `_C` compiled module.
* **The Fix:** Renamed the local repo directory to prevent shadowing. We ignored the missing `_C` binary because it only handles mask hole-filling post-processing, which we don't need for basic DINOv2 bounding box cropping.

---

## 3. Architectural Streamlining

### Hurdle: VRAM Duplication (The OOM Threat)
* **The Problem:** The legacy SAM6D code built a `SAM2ImagePredictor` for the AMG phase and a separate `SAM2VideoPredictor` for tracking. This loaded the massive Transformer backbone into VRAM twice.
* **The Engineering Decision:** Build the `sam2_video_predictor` once and pass it as the underlying model to the AMG.
    ```python
    # Single model loaded into VRAM
    seg_tracker = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=DEVICE).to(torch.bfloat16)
    seg_generator = SAM2AutomaticMaskGenerator(model=seg_tracker, ...)
    ```

### Hurdle: Video Predictors Can't Read Live Webcams
* **The Problem:** SAM 2 natively requires a path to a folder of pre-saved `.jpg` frames (`init_state`), causing immediate RAM explosion on live infinite loops.
* **The Engineering Decision:** Created `SAM2CameraStreamWrapper`. It intercepts the `inference_state` dictionary, accepts live NumPy arrays, and manually garbage-collects old frames to keep VRAM perfectly flat.
    ```python
    # Garbage collection preventing memory leaks
    def _clear_old_memory(self):
        old_frame_idx = self.frame_idx - self.max_memory_frames
        if old_frame_idx in self.inference_state["images"]:
            del self.inference_state["images"][old_frame_idx]
    ```

---

## 4. The Memory Battles (Slaying `NvMapMem error 12`)

### Hurdle: Float32 Math Crashing the Unified Memory
* **The Problem:** PyTorch defaulted to FP32, which disabled FlashAttention and forced $O(N^2)$ memory allocation for the Transformer attention blocks, instantly crashing the 8GB Jetson.
* **The Engineering Decision:** Forced the entire pipeline (model weights and intermediate activations) into `bfloat16` using `.to()` and `torch.autocast`.
    ```python
    # Forcing mixed precision for math operations
    with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
    ```

### Hurdle: AMG Generating Too Many Proposals
* **The Problem:** Default AMG puts a 32x32 grid (1024 points) over the image. Feeding 1024 masks through the decoder and then into DINOv2 caused OOM errors.
* **The Engineering Decision:** Neuter the AMG. Drop points to 16x16 (256 points) and process them in batches of 16.
    ```python
    # Neuter AMG to protect Jetson VRAM
    points_per_side=16,
    points_per_batch=16, 
    ```

### Hurdle: DINOv2 Batching OOM
* **The Problem:** Throwing even 40 masks into DINOv2 at once spiked VRAM.
* **The Engineering Decision:** Loop and chunk the `forward_features` pass.
    ```python
    # DINOv2 Chunking
    chunk_size = 16
    for start_idx in range(0, N, chunk_size):
        chunk = batch_crops[start_idx:end_idx].to(torch.bfloat16)
    ```

---

## 5. The Sim2Real & Mathematical Bugs

### Hurdle: The "Gray Background" Domain Gap
* **The Problem:** DINOv2 was fed crops where the background was gray instead of black. We applied ImageNet normalization *before* applying the boolean mask. `(0.0 - mean) / std` turned the `[0,0,0]` black background into a negative number, which rendered as the ImageNet mean (gray).
* **The Engineering Decision:** Mask the image *first* while it is still in the `[0, 1]` range, ensuring the background is mathematically `0.0`, then normalize right before inference.
    ```python
    # Apply mask FIRST to maintain true black [0,0,0]
    masked_crop = crop_img * crop_mask
    ...
    # Normalize LAST
    batch_crops = self.normalize(batch_crops)
    ```

### Hurdle: Sim2Real Texture Mismatch
* **The Problem:** Comparing a real, black, textured MG996R servo against a flat, white, untextured BlenderProc CAD mesh resulted in Cosine Similarities peaking at ~0.59.
* **The Engineering Decision:** Lowered the `acquisition_threshold`. Since DINOv2 was successfully finding the 3D geometry despite the color mismatch, 0.50 was a safe lock-on point.
    ```python
    # Lowered threshold for untextured CAD matching
    self.acquisition_threshold = 0.5 
    ```

---

## 6. Edge Deployment Realities

### Hurdle: Headless SSH Rendering
* **The Problem:** Calling `cv2.imshow` over an SSH connection without X11 forwarding immediately crashes the script.
* **The Engineering Decision:** Replace live UI rendering with a continuous `cv2.VideoWriter` that saves an `.mp4` file, and implement a graceful `try/except KeyboardInterrupt` loop.
    ```python
    # Jetson headless video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('tracking_output.mp4', fourcc, fps_out, ...)
    ```

### Hurdle: The CPU/GPU Memory Boundary
* **The Problem:** Attempting to slice a CPU OpenCV NumPy frame using a GPU PyTorch boolean mask tensor threw a `TypeError`.
* **The Engineering Decision:** Bulletproof visualization by explicitly pulling masks back to host RAM before NumPy operations.
    ```python
    # Crossing the device boundary safely
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    ```