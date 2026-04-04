# download_models.py

import os
import subprocess

def download_file(command_list, cwd=None):
    """Run a shell command for downloading files."""
    try:
        subprocess.run(command_list, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing: {' '.join(command_list)}\n{e}")

# Instance Segmentation Model
instance_model_dir = "ism"
os.makedirs(instance_model_dir, exist_ok=True)

# SAM
sam_checkpoint_dir = os.path.join(instance_model_dir, "checkpoints", "segment-anything")
os.makedirs(sam_checkpoint_dir, exist_ok=True)
download_file([
    "wget", "-P", sam_checkpoint_dir,
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "--no-check-certificate"
])

# DinoV2
dinov2_checkpoint_dir = os.path.join(instance_model_dir, "checkpoints", "dinov2")
os.makedirs(dinov2_checkpoint_dir, exist_ok=True)
download_file([
    "wget", "-P", dinov2_checkpoint_dir,
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "--no-check-certificate"
])

# Pose Estimation Model
pose_model_dir = "Pose_Estimation_Model"
os.makedirs(os.path.join(pose_model_dir, "checkpoints"), exist_ok=True)
download_file([
    "gdown", "--no-cookies", "--no-check-certificate",
    "-O", os.path.join(pose_model_dir, "checkpoints", "sam-6d-pem-base.pth"),
    "1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7"
])
