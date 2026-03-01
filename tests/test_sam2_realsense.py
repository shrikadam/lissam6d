import os
import time
import cv2
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image
from sam2.build_sam import build_sam2_object_tracker
import pyrealsense2 as rs 
import cv2
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image

from sam2.build_sam import build_sam2_auto_mask_generator, build_sam2_object_tracker

# List to store the clicked points
objects = []
points = []

# Mouse callback function
def register_bb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")
            # Optional: draw a circle on the clicked point
            cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("First Frame", color_image)

class Visualizer:
    def __init__(self,
                 video_width,
                 video_height,
                 ):
        
        self.video_width = video_width
        self.video_height = video_height

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False,
                                               )
        
        return mask

    def draw_seg_mask(self, frame, mask):
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()
        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            frame[obj_mask] = [255, 105, 180]

        return frame
    
    def add_frame(self, frame, mask):
        frame = self.draw_seg_mask(frame, mask)
        rgb_frame = Image.fromarray(frame)
        clear_output(wait=True)
        display(rgb_frame)

# Set SAM2 Configuration
NUM_OBJECTS = 2
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
DEVICE = 'cuda:0'

# Open Realsense Video Stream
pipe = rs.pipeline()
cfg = rs.config()
aligner = rs.align(rs.stream.color)
W, H = 1280, 720
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
pipe.start(cfg)

# For real-time visualization
visualizer = Visualizer(video_width=W, video_height=H)

sam_tracker = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False
                                )

available_slots = np.inf
first_frame = True

# BB Registration
cv2.namedWindow('First Frame')
cv2.setMouseCallback('First Frame', register_bb)
instruction_text = "Click two points on the image to select the desired object bounding box."
warning_text = "!!!ENSURE TO KEEP THE CAMERA AND OBJECT STEADY DURING THIS STEP!!!"

with torch.inference_mode(), torch.autocast('cuda:0', dtype=torch.bfloat16):
    while True:
        start_time = time.time()

        # Get next frame
        frame_ = pipe.wait_for_frames()
        frame = aligner.process(frame_)
        color_frame = frame.get_color_frame()
        depth_frame = frame.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_buffer, cv2.COLOR_RGB2BGR)
        # Get target bbox on a static frame
        if first_frame:
            cv2.putText(color_image, instruction_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (57, 255, 20), 1, cv2.LINE_AA)
            cv2.putText(color_image, warning_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            while True:
                cv2.imshow("First Frame", color_image)
                key = cv2.waitKey(20) & 0xFF
                # Exit loop if 'Esc', 'Enter', or any key is pressed OR two points are clicked
                if key != 255 or len(points) == 2:
                    cv2.destroyAllWindows()
                    break
            bbox = np.array([points])
            sam_out = sam_tracker.track_new_object(img=color_image, box=bbox)
            first_frame = False 
        else:
            sam_out = sam_tracker.track_all_objects(img=color_image)
            
        seg_mask = visualizer.draw_seg_mask(frame=color_image, mask=sam_out['pred_masks'])
        cv2.imshow("SAM2 Realtime", seg_mask)
        if cv2.waitKey(1) == ord('q'):
            break
        
pipe.stop()
cv2.destroyAllWindows()