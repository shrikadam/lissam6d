## Model-based realtime 6D detection with RGB-D camera

### 1. Environment Setup
CUDA versions should match between GPU Driver, CUDA Toolkit and PyTorch.
```shell
conda create -n sam6d python=3.10
conda activate sam6d
conda install cuda=12.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Build Pointnet++ extensions for PEM.
```shell
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
```
Download SAM, Dino V2 and PEM weights.
```shell
python download_weights.py
```

### 2. Evaluation on the camera stream

#### Run the template render (one-time)
```shell
blenderproc run render_templates.py --object bolt
```

#### Run the inference
```shell
python run_inference_camera.py --object bolt
```

## Citation
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, 
  Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, 
  Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, 
  Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@misc{yang2024samurai,
      title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
      author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
      year={2024},
      eprint={2411.11922},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.11922}, 
}
```
---