import gorilla
import argparse
import os
import sys
from PIL import Image
import os.path as osp
import numpy as np
import random
import importlib
import json

import torch
import torchvision.transforms as transforms
import cv2
from rich.progress import Progress

import pycocotools.mask as cocomask
import trimesh

from utils.data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from utils.draw_utils import draw_detections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

def load_json(path):
    with open(path, "r") as f:
        info = json.load(f)
    return info

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

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

            vis_img = visualize(img, pred_rot, pred_trans, model_points*1000, cam_K, save_path)
            vis_img.save(save_path)

def fuse_pem_json(output_dir):
    combined_data = []
    for file_name in os.listdir(output_dir):
        # Only consider JSON files matching the pattern
        if file_name.startswith('detection_pem_img') and file_name.endswith('.json'):
            # Full path to the current JSON file
            json_file_path = os.path.join(output_dir, file_name)
            # Read and load the contents of the current JSON file
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                combined_data.extend(data)
    output_path = os.path.join(output_dir, "detection_pem.json")
    # Write fused data
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"Fused ISM JSON files into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Estimation")
    # pem
    parser.add_argument("--gpus", type=str,  default="0", help="path to pretrain model")
    parser.add_argument("--model",  type=str, default="pose_estimation_model", help="path to model file")
    parser.add_argument("--config", type=str,  default="config/base.yaml", help="path to config file, different config.yaml use different config")
    parser.add_argument("--iter", type=int, default=600000, help="epoch num. for testing")
    parser.add_argument("--exp_id", type=int, default=0, help="")
    # input
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--input_dir", nargs="?", help="Path to root directory of the input")
    parser.add_argument("--template_dir", nargs="?", help="Path to templates")
    parser.add_argument("--cad_dir", nargs="?", help="Path to CAD folder")
    parser.add_argument("--targets_path", default='?', nargs="?", help="Path to test/val targets json")
    parser.add_argument("--detection_path", nargs="?", help="Path to segmentation information(generated by ISM)")
    parser.add_argument("--det_score_thresh", default=0.4, help="The score threshold of detection")
    args = parser.parse_args()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus     = args.gpus
    cfg.model_name = args.model
    cfg.log_dir  = log_dir
    cfg.test_iter = args.iter
    cfg.det_score_thresh = args.det_score_thresh
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    print("Initializing model...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    model = model.cuda()
    model.eval()
    checkpoint = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'checkpoints', 'sam-6d-pem-base.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    input_folders = sorted(os.listdir(args.input_dir))
    with Progress() as progress:
        input_tqdm = progress.add_task('input_folders', total=len(input_folders))    
        for input_folder in input_folders:
            input_dir = os.path.join(args.input_dir, input_folder)
            output_dir = os.path.join(args.output_dir, input_folder)
            run_inference(model, cfg, output_dir, input_dir, args.targets_path, args.detection_path, args.template_dir, args.cad_dir)
            progress.update(input_tqdm, advance=1)
            fuse_pem_json(output_dir)
