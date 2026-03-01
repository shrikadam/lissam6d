import os
import gc
import importlib
import numpy as np
import torch
from PIL import Image
import logging
import gorilla
from hydra import initialize, compose
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
import random
from hydra.utils import instantiate
import argparse
import glob
from rich.progress import Progress
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import cv2
import imageio.v2 as imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import pyrealsense2 as rs 
import json
import keyboard as kb

from ism.utils.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from ism.utils.bbox_utils import CropResizePad
from ism.model.utils import Detections, convert_dets_to_json
from pem.utils.data_utils import load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose
from pem.utils.draw_utils import draw_detections


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

def visualize_pem(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    vis_pem = Image.fromarray(np.uint8(img))

    return vis_pem

def batch_camera_data(cam_path, device):
    cam_batched = {}
    with open(cam_path, "r") as f:
        cam_info = json.load(f)
    cam = cam_info[next(iter(cam_info))]
    cam_K = np.array(cam['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam['depth_scale'])
    cam_batched["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    cam_batched['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return cam_batched

def _get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz

def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose

def get_detections(detections, scene_id, image_id, obj_id):
    dets_ = []
    for det in detections:
        if det['scene_id'] == scene_id and det['image_id'] == image_id and det['category_id'] == obj_id:
            dets_.append(det)
    return dets_

def get_test_data(input_dir, cad_dir, det_score_thresh, cfg, detections, scene_id, im_id, obj_id):
    rgb_path = os.path.join(input_dir, "rgb_cam1", f"{im_id:06d}.png")
    depth_path = os.path.join(input_dir, "depth_cam1", f"{im_id:06d}.png")
    cam_path = os.path.join(input_dir, "scene_camera_cam1.json")
    cad_path = os.path.join(cad_dir, f"obj_{obj_id:06d}.ply")

    dets_ = get_detections(detections, scene_id, im_id, obj_id)
    assert len(dets_) > 0
    dets = []
    for det in dets_:
        if det['score'] > det_score_thresh:
            dets.append(det)
    del dets_, detections

    cam_info = load_json(cam_path)
    cam = cam_info[next(iter(cam_info))]
    K = np.array(cam['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam['depth_scale'])

    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * depth_scale / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']
        # Mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]
        # Points
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]
        # RGB
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    if all_rgb == []:
        return None, None, None, None, None
    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets

def run_inference(model, cfg, output_dir, input_dir, test_targets_path, ism_detection_path, template_dir, cad_dir):
    with open (test_targets_path, "r") as f:    
        test_targets = json.load(f)

    det_score_thresh = float(cfg.det_score_thresh)
    scene_str = os.path.basename(input_dir)
    scene_id = int(scene_str)
    im_idx = [item['im_id'] for item in test_targets if item['scene_id']==scene_id]
    img_files = [f for f in os.listdir(os.path.join(input_dir, "rgb_cam1")) if f.endswith('.png')]
    im_idx = im_idx[:len(img_files)]

    for im_id in im_idx:
        res = []
        ism_detection_json = os.path.join(os.path.join(ism_detection_path, scene_str, f"detection_ism.json"))
        with open(ism_detection_json, "r") as f:
            detection_masks = json.load(f)
        det_masks = [item for item in detection_masks if item['scene_id'] == scene_id and item['image_id'] == im_id and item['score'] > det_score_thresh]
        if len(det_masks) == 0:  
            print(f"No Segmentation masks found for scene {scene_id}: image {im_id}!")           
            continue
        det_obj_ids = set([item['category_id'] for item in det_masks])
        for obj_id in det_obj_ids:
            tem_path = os.path.join(template_dir, "obj_{:06d}".format(obj_id))
            all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, cfg.test_dataset)
            with torch.no_grad():
                all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

            input_data, img, whole_pts, model_points, detections = get_test_data(
                input_dir, cad_dir, det_score_thresh, cfg.test_dataset, 
                detection_masks, scene_id, im_id, obj_id
            )
            if input_data is None:
                print(f"No input data found for scene {scene_id}: image {im_id}!")
                continue
            ninstance = input_data['pts'].size(0)
            
            with torch.no_grad():
                input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
                input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
                
                input_keys = ['pts', 'rgb', 'rgb_choose', 'score', 'model', 'K', 'dense_po', 'dense_fo']
                output_keys = ['pts', 'rgb', 'rgb_choose', 'score', 'model', 'K', 'dense_po', 'dense_fo', 'init_R', 'init_t', 'pred_R', 'pred_t', 'pred_pose_score']
                out = {}
                for key in output_keys:
                    out[key] = []
                for idx in range(len(detections)):
                    current_data = {key: input_data[key][idx:idx+1] for key in input_keys}
                    current_out = model(current_data)
                    for key in output_keys:
                        out[key].append(current_out[key])
                for key in output_keys:
                    out[key] = torch.cat(out[key], dim=0)
                    

            if 'pred_pose_score' in out.keys():
                pose_scores = out['pred_pose_score'] * out['score']
            else:
                pose_scores = out['score']
            pose_scores = pose_scores.detach().cpu().numpy()
            pred_rot = out['pred_R'].detach().cpu().numpy()
            pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

            os.makedirs(f"{output_dir}", exist_ok=True)
            for idx, det in enumerate(detections):
                detections[idx]['score'] = float(pose_scores[idx])
                detections[idx]['R'] = list(pred_rot[idx].tolist())
                detections[idx]['t'] = list(pred_trans[idx].tolist())
            for det in detections:
                res.append({
                    "scene_id": str(scene_id),
                    "im_id": str(im_id),
                    "obj_id": str(obj_id),
                    "score": det["score"],
                    "R": det["R"],
                    "t": det["t"],
                    "time": det["time"],
                })            

            with open(os.path.join(f"{output_dir}", f'detection_pem_img{im_id}_obj{obj_id}.json'), "w") as f:
                json.dump(res, f, indent=2)

        image_path = os.path.join(input_dir, "rgb_cam1", f"{im_id:06d}.png")
        img = load_im(image_path).astype(np.uint8)
        if len(img.shape)==2:
            img = np.concatenate([img[:,:,None], img[:,:,None], img[:,:,None]], axis=2)

        camera_path = os.path.join(input_dir, "scene_camera_cam1.json")
        cam_info = load_json(camera_path)
        cam = cam_info[next(iter(cam_info))]

        det_obj_ids = set([int(item['obj_id']) for item in res])
        for obj_id in det_obj_ids:
            save_path = os.path.join(f"{output_dir}", f'vis_pem_img{im_id}_obj{obj_id}.png')

            cad_path = os.path.join(cad_dir, f"obj_{obj_id:06d}.ply")
            mesh = trimesh.load_mesh(cad_path)
            model_points = mesh.sample(cfg.test_dataset.n_sample_model_point).astype(np.float32) / 1000.0

            pred_rot = np.array([item["R"] for item in res if item["obj_id"] == str(obj_id)])
            pred_trans = np.array([item["t"] for item in res if item["obj_id"] == str(obj_id)])
            cam_K = np.array(cam['cam_K']).reshape((1, 3, 3)).repeat(pred_rot.shape[0], axis=0)

            vis_img = visualize_pem(img, pred_rot, pred_trans, model_points*1000, cam_K, save_path)
            vis_img.save(save_path)

def infer_pose(seg_model, pem_model, color_image, depth_image, batch, obj_id, mesh, visualize=False):
    # generate masks
    rgb = Image.fromarray(color_image)
    detections = seg_model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = seg_model.descriptor_model.forward(np.array(rgb), detections)
    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = seg_model.compute_semantic_score(query_decriptors)
    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
    # compute the appearance score
    appe_scores, ref_aux_descriptor= seg_model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    depth = depth_image.astype(np.int32)
    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    # compute the geometric score
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4 # type: ignore
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    seg_model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    seg_model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    image_uv = seg_model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = seg_model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=seg_model.visible_thred
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
    parser.add_argument("--det_score_thresh", default=0.4, help="The score threshold of detection")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ######################## Initialize Segmentation Model ########################
    
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
    seg_model = instantiate(cfg.model)
    
    seg_model.descriptor_model.model = seg_model.descriptor_model.model.to(device)
    seg_model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(seg_model.segmentor_model, "predictor"):
        seg_model.segmentor_model.predictor.model = (
            seg_model.segmentor_model.predictor.model.to(device)
        )
    else:
        seg_model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
    
    ######################## Initialize Pose Estimation Model ########################

    cfg = gorilla.Config.fromfile("pem/config/base.yaml")
    cfg.det_score_thresh = args.det_score_thresh
    gorilla.utils.set_cuda_visible_devices(gpu_ids = "0")

    print("Initializing pose estimation model...")
    PEM_MODEL = importlib.import_module("pose_estimation_model")
    pem_model = PEM_MODEL.Net(cfg.model)
    pem_model = pem_model.cuda()
    pem_model.eval()
    checkpoint = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'checkpoints', 'sam-6d-pem-base.pth')
    gorilla.solver.load_checkpoint(model=pem_model, filename=checkpoint)

    ######################## Prepare Reference Data for Model ########################

    cam_path = os.path.join(base_dir, f'data/cam.json')
    cam_batched = batch_camera_data(cam_path, device)

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

    seg_model.ref_data = {}
    seg_model.ref_data["descriptors"] = seg_model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    seg_model.ref_data["appe_descriptors"] = seg_model.descriptor_model.compute_masked_patch_feature(
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
        infer_pose(seg_model, pem_model, color_image, depth_image, cam_batched, obj_id, mesh, visualize=False)

        if kb.is_pressed('esc'):
            print("Esc pressed, exiting inference...")
            break

    pipe.stop()