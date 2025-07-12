import os
import gc
import numpy as np
import torch
from PIL import Image
import logging
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import cv2
import imageio.v2 as imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from ism.utils.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from ism.utils.bbox_utils import CropResizePad
from ism.model.utils import Detections, convert_dets_to_json
import json
import keyboard as kb
import time
import pyrealsense2 as rs 

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

def visualize_ism(rgb, detections):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33
    
    for det_id, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        r = int(255*colors[det_id][0])
        g = int(255*colors[det_id][1])
        b = int(255*colors[det_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255

    vis_ism = Image.fromarray(np.uint8(img))

    return vis_ism

def batch_camera_data(cam_path, device):
    batch = {}
    with open(cam_path, "r") as f:
        cam_info = json.load(f)
    cam = cam_info[next(iter(cam_info))]
    cam_K = np.array(cam['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam['depth_scale'])
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def infer_pose(model, color_image, depth_image, batch, obj_id, mesh, visualize=False):
    # generate masks
    rgb = Image.fromarray(color_image)
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)
    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)
    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    depth = depth_image.astype(np.int32)
    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    # compute the geometric score
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4 # type: ignore
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )
    
    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.full_like(final_score, obj_id)) 
    detections.to_numpy()
    detection_results = detections.save_to_file(0, 0, 0, 'tmp', return_results=True)
    detection_json = convert_dets_to_json(detection_results)
    vis_ism = visualize_ism(rgb, detection_json)
    vis_ism_cv = cv2.cvtColor(np.array(vis_ism), cv2.COLOR_RGB2BGR)
    cv2.imshow("ISM Result", vis_ism_cv)
    cv2.waitKey(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--object", default='bolt', nargs="?", help="Object name")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with initialize(version_base=None, config_path="ism/configs"):
        cfg = compose(config_name='run_inference.yaml')

    segmentor_model = args.segmentor_model
    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="ism/configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = args.stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="ism/configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    ######################## Prepare Reference Data for Model ########################

    cam_path = os.path.join(base_dir, f'data/cam.json')
    batch = batch_camera_data(cam_path, device)

    ######################## Prepare Reference Data for Model ########################
    cad_path = os.path.join(base_dir, f'data/{args.object}/{args.object}.ply')
    template_path = os.path.join(base_dir, f'data/{args.object}/templates')

    mesh = trimesh.load_mesh(cad_path)
    num_templates = len(glob.glob(f"{template_path}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_path, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_path, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())
        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = [box for box in boxes if box is not None]
    boxes = torch.tensor(np.array(boxes))

    processing_config = OmegaConf.create({"image_size": 224})
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data

    ######################## Set up Realsense Camera ########################

    pipe = rs.pipeline()
    cfg = rs.config()
    aligner = rs.align(rs.stream.color)

    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipe.start(cfg)

    obj_id = 0

    while True:
        frame_ = pipe.wait_for_frames()
        frame = aligner.process(frame_)
        color_frame = frame.get_color_frame()
        depth_frame = frame.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        infer_pose(model, color_image, depth_image, batch, obj_id, mesh, visualize=False)

        if kb.is_pressed('esc'):
            print("Esc pressed, exiting inference...")
            break

    pipe.stop()