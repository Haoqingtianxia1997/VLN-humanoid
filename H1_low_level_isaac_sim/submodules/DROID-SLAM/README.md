# DROID-SLAM

## Requirements

To run the code you will need ...
* **Inference:** Running the demos will require a GPU with at least 11G of memory. 

* **Training:** Training requires a GPU with at least 24G of memory. We train on 4 x RTX-3090 GPUs.

### Installing

Requires CUDA to be installed on your machine. If you run into issues, make sure the PyTorch and CUDA major versions match with the following check (minor version mismatch should be fine).

### Dependencies

1. nvidia cuda toolkit 12.6(only tested this one, but 12.x should all work)
Link: https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

2. python 3.10 virtual environment
```bash
sudo apt install python3.10-venv
```

#### add the following to your ~/.bashrc:
```bash
# nvidia-cuda-toolkit from apt repository
#export CUDA_HOME=/usr
#export PATH=/usr/bin:$PATH
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# cuda toolkit 12.6, comment if default 11.5 needed
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### check if cuda version is correct after source
```bash
which nvcc
nvcc -V
python3 -m venv .venv
source .venv/bin/activate
```
```Bash
# install requirements (tested up to torch 2.7)
pip install -r requirements.txt
python -c "import torch; print(torch.version.cuda)"

# optional (for visualization)
pip install moderngl moderngl-window

# install third-party modules (this will take a while)
# in droid-slam main folder
git submodule update --init --recursive .
# submodule lietorch
cd thirdparty/lietorch
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"
pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git
pip install opencv-python open3d scipy pyyaml
# submodule pytorch_scatter
cd ../..
pip install thirdparty/pytorch_scatter

# install droid-backends
pip install -e .
# optional: if droid-backends installation fails:
    sudo apt update
    sudo apt install -y libeigen3-dev ninja-build 
    # then expose Eigen's include path to the compiler and install droid-backends again
    export CPATH=/usr/include/eigen3:$CPATH
    export CPLUS_INCLUDE_PATH=/usr/include/eigen3:$CPLUS_INCLUDE_PATH
    pip install -e .

# download model
./tools/download_model.sh
```

## Project related demo
```bash
# change the path for rgb and depth accordingly, see vln-humanoid repo
python3 demo.py --imagedir=$(vln-humanoids-home)/camera_feed/legged/rgb --depthdir=$(vln-humanoids-home)/camera_feed/legged/depth --calib=calib/h1.txt --stride 1 --upsample

# subscribe ros2 topic
source /opt/ros/humble/setup.bash

python3 demo_ros2_realtime.py --rgb_topic=/camera/rgb/image_raw --depth_topic=/camera/depth/image_raw --calib=calib/h1.txt --upsample --asynchronous --publish_pose

# optional: test droid-slam performance based on the jetbot in vln_humanoid repo
python3 demo.py --imagedir=$(vln-humanoids-home)/camera_feed/wheeled/rgb --depthdir=$(vln-humanoids-home)/camera_feed/wheeled/depth --calib=calib/wheeled_robot.txt
```




































## Demos(from original droid-slam as reference)

1. Download the model from google drive: [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing) or with
    ```Bash
    ./tools/download_model.sh
    ```

2. Download some sample videos using the provided script.
    ```Bash
    ./tools/download_sample_data.sh
    ```

Run the demo on any of the samples (all demos can be run on a GPU with 11G of memory). To save the reconstruction with full resolution depth maps use the `--reconstruction_path` flag. If you ran with `--reconstruction_path my_reconstruction.pth`, you can view the reconstruction in high resolution by running
```Bash
python view_reconstruction.py my_reconstruction.pth
```

**Asynchronous and Multi-GPU Inference:** You can run the demos in asynchronous mode by running with `--asynchronous`. In this setting, the frontend and backend will run in seperate Python processes. You can additionally enable multi-GPU inference by setting the devices of the frontend and backend processes with the following arguments. 

**Visualization currently doesn't work multi-gpu setting**. You will need to run with ``--disable_vis``.


```Bash
python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt
```

```Bash
python demo.py --imagedir=data/mav0/cam0/data --calib=calib/euroc.txt --t0=150
```

```Bash
python demo.py --imagedir=data/rgbd_dataset_freiburg3_cabinet/rgb --calib=calib/tum3.txt
```


**Running on your own data:** All you need is a calibration file. Calibration files are in the form 
```
fx fy cx cy [k1 k2 p1 p2 [ k3 [ k4 k5 k6 ]]]
```
with parameters in brackets optional.

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, TUM, and ETH3D-SLAM. EuRoC and TUM can be run on a 1080Ti. The TartanAir and ETH3D-SLAM datasets will require 24G of memory. 

**Asynchronous and Multi-GPU Inference:** You can run evaluation in asynchronous mode by running with `--asynchronous`. In this setting, the frontend and backend will run in seperate Python processes. You can additionally enable multi-GPU inference by setting the devices of the frontend and backend processes with the following arguments. For example:
```
python evaluation_scripts/test_tartanair.py \
  --datapath data/tartanair_test/mono \
  --gt_path data/tartanair_test/mono_gt \
  --frontend_device cuda:0 \
  --backend_device cuda:1 \
  --asynchronous \
  --disable_vis
```


**Note:** Running with `--asynchronous` will typically produce better results, but this mode is not deterministic.

### TartanAir (Mono + Stereo)

Download the [TartanAir](https://theairlab.org/tartanair-dataset/) test set with this command.

```Bash
./tools/download_tartanair_test.sh
```

Or from these links: [Images](https://drive.google.com/file/d/1N8qoU-oEjRKdaKSrHPWA-xsnRtofR_jJ/view), [Groundtruth](https://cmu.box.com/shared/static/3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip) 



**Monocular evaluation:**
```bash
python evaluation_scripts/test_tartanair.py \
  --datapath datasets/tartanair_test/mono \
  --gt_path datasets/tartanair_test/mono_gt \
  --disable_vis
```

**Stereo evaluation:**
```bash
python evaluation_scripts/test_tartanair.py \
  --datapath datasets/tartanair_test/stereo \
  --gt_path datasets/tartanair_test/stereo_gt \
  --stereo --disable_vis
```


**Evaluating on the validation split:**

Download the [TartanAir](https://theairlab.org/tartanair-dataset/) dataset using the script `thirdparty/tartanair_tools/download_training.py` and put them in `datasets/TartanAir`
```Bash
# monocular eval
./tools/validate_tartanair.sh --plot_curve

# stereo eval
./tools/validate_tartanair.sh --plot_curve  --stereo
```

### EuRoC (Mono + Stereo)
Download the [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) sequences (ASL format):
```Bash
./tools/download_euroc.sh
```

Then run evaluation:
```Bash
# monocular eval (single gpu)
./tools/evaluate_euroc.sh

# monocular eval (multi gpu)
./tools/evaluate_euroc.sh --asynchronous --frontend_device cuda:0 --backend_device cuda:1

# stereo eval (single gpu)
./tools/evaluate_euroc.sh --stereo

# stereo eval (multi gpu)
./tools/evaluate_euroc.sh --stereo --asynchronous --frontend_device cuda:0 --backend_device cuda:1
```

### TUM-RGBD (Mono)
Download the [TUM-RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) sequences:
```
./tools/download_tum.sh
```
Then run evaluation:
```Bash
# monocular eval (single gpu)
./tools/evaluate_tum.sh

# monocular eval (multi gpu)
./tools/evaluate_tum.sh --asynchronous --frontend_device cuda:0 --backend_device cuda:1
```

### ETH3D (RGB-D)
Download the [ETH3D](https://www.eth3d.net/slam_datasets) dataset:
```Bash
./tools/download_eth3d.sh
```

```Bash
# RGB-D eval (single gpu)
./tools/evaluate_eth3d.sh > eth3d_results.txt
python evaluation_scripts/parse_results.py eth3d_results.txt

# RGB-D eval (multi gpu)
./tools/evaluate_eth3d.sh --asynchronous --frontend_device cuda:0 --backend_device cuda:1 > eth3d_results_async.txt
python evaluation_scripts/parse_results.py eth3d_results_async.txt
```

## Training

First download the TartanAir dataset. The download script can be found in `thirdparty/tartanair_tools/download_training.py`. You will only need the `rgb` and `depth` data.

```
python download_training.py --rgb --depth
```

You can then run the training script. We use 4x3090 RTX GPUs for training which takes approximatly 1 week. If you use a different number of GPUs, adjust the learning rate accordingly.

**Note:** On the first training run, covisibility is computed between all pairs of frames. This can take several hours, but the results are cached so that future training runs will start immediately. 


```
python train.py --datapath=<path to tartanair> --gpus=4 --lr=0.00025
```


## Acknowledgements
Data from [TartanAir](https://theairlab.org/tartanair-dataset/) was used to train our model. We additionally use evaluation tools from [evo](https://github.com/MichaelGrupp/evo) and [tartanair_tools](https://github.com/castacks/tartanair_tools).
