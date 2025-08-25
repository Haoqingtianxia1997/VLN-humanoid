# Usage guide of visual language navigation for Unitree H1 humanoid

Check README in each subfolder and install all the dependencies.

No further git clone and other git commands for submodules needed. Everything is already included.

48GB of VRAM is needed to run the total system.

## 1. launch Isaac Sim
```bash
conda activate env_isaaclab # assume it's your conda env's name for isaac sim&lab
source /opt/ros/humble/setup.bash
cd H1_low_level_isaac_sim
./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_udp.py --infinite-episode --enable_cameras --udp --checkpoint policy/policy_rough.pt
```

## 2. launch DROID SLAM
```bash
cd DROID-SLAM
source .venv/bin/activate
python3 demo_ros2_realtime.py --rgb_topic=/camera/rgb/image_raw --depth_topic=/camera/depth/image_raw --calib=calib/h1.txt --upsample --asynchronous --publish_pose
```

## 3. Explore the environment manually with udp command
```bash
cd H1_low_level_isaac_sim
python3 scripts/utils/udp/udp_keyboard_control_humanoid.py
```

## 4. High level part for VLM&LLM
```bash
cd H1_high_level_isaac_sim
conda activate H1_Nav # assume it's your conda env's name for h1 high level, check requirements.txt
python3 main.py
```