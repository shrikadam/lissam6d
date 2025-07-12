import pyrealsense2 as rs 
import numpy as np
import cv2
from ultralytics import SAM

pipe = rs.pipeline()
cfg = rs.config()
aligner = rs.align(rs.stream.color)

cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

pipe.start(cfg)

seg_model = SAM("sam2.1_b.pt")

while True:
    frame_ = pipe.wait_for_frames()
    frame = aligner.process(frame_)
    color_frame = frame.get_color_frame()
    depth_frame = frame.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    res = seg_model.predict(color_image)[0]
    H, W = color_image.shape[:2]
    for r in res:
        img = np.copy(r.orig_img)
        for ci, c in enumerate(r):
            mask = np.zeros((H, W), np.uint8)
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            seg_vis = cv2.drawContours(color_image, [contour], -1, (255, 255, 255), cv2.FILLED)
    # depth_image = np.asanyarray(depth_frame.get_data())
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.5), cv2.COLORMAP_JET)
    # color_mat = cv2.Mat(color_image)
    # depth_mat = cv2.Mat(depth_image)
    # cv2.imshow("Color Image", color_image)
    # cv2.imshow("Depth Image", depth_colormap)
    cv2.imshow("Segmentation Results", seg_vis)
    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()