# Description

# Preliminaries
1. Ros2 Humble
2. Isaac Sim 4.5, link: https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html
3.  Isaac Lab(pip), link: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

# Installation guide
```bash
git clone git@gitlab.com:telekinesis/vln-humanoids.git
git checkout master
git submodule update --init --recursive .
```
# Note: check README in each submodule except isaaclab

# conda environment
Use your env_isaaclab from the installation guide of isaaclab

```bash
conda activate env_isaaclab
pip install -r requirements.txt

# source ros2 humble in each terminal
source /opt/ros/humble/setup.bash
```

## GUIDE
run the following commands in vln-humanoids main folder
#### train
```bash
./submodules/isaaclab/isaaclab.sh -p submodules/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-H1-v0 \
    --headless \
    --max_iterations 10000 \
    --num_envs 1024 \
    env.commands.base_velocity.ranges.lin_vel_x=[-1.0,1.0] \
    env.commands.base_velocity.ranges.lin_vel_y=[-1.0,1.0] \
    env.commands.base_velocity.ranges.ang_vel_z=[-1.0,1.0] \
    env.rewards.track_lin_vel_xy_exp.weight=40.0 \
    env.rewards.track_ang_vel_z_exp.weight=50.0

./submodules/isaaclab/isaaclab.sh -p submodules/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-H1-v0 \
    --headless \
    --max_iterations 10000 \
    --num_envs 1024 \
    env.commands.base_velocity.ranges.lin_vel_x=[-1.0,1.0] \
    env.commands.base_velocity.ranges.lin_vel_y=[-1.0,1.0] \
    env.commands.base_velocity.ranges.ang_vel_z=[-1.0,1.0] 
```

#### play
```bash
./submodules/isaaclab/isaaclab.sh -p submodules/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-H1-Play-v0 \
  --checkpoint logs/rsl_rl/h1_rough/2025-07-28_04-30-48/model_9999.pt \
  --num_envs 1024 \
  env.commands.base_velocity.ranges.lin_vel_x=[-1.0,1.0] \
  env.commands.base_velocity.ranges.lin_vel_y=[-1.0,1.0] \
  env.commands.base_velocity.ranges.ang_vel_z=[-1.0,1.0]
# or
./submodules/isaaclab/isaaclab.sh -p submodules/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-H1-Play-v0 \
    --checkpoint logs/rsl_rl/h1_flat/2025-07-29_03-50-55/model_9999.pt \
    --num_envs 1024 \
    env.commands.base_velocity.ranges.lin_vel_x=[-1.0,1.0] \
    env.commands.base_velocity.ranges.lin_vel_y=[-1.0,1.0] \
    env.commands.base_velocity.ranges.ang_vel_z=[-1.0,1.0]
```

#### inference with udp control
```bash
# First terminal
source /opt/ros/humble/setup.bash

./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_udp.py --infinite-episode --enable_cameras --udp --checkpoint policy/policy_rough.pt
# or
./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_udp.py --infinite-episode --enable_cameras --udp --checkpoint policy/policy_flat.pt # --udp-host 0.0.0.0 --udp-port 8888
# or mpc with rough terrain
./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_udp.py --infinite-episode --enable_cameras --udp --checkpoint policy/policy_rough.pt --enable-trajectory  --trajectory-scale 5.0
# or mpc with flat terrain
# ./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_udp.py --infinite-episode --enable_cameras --udp --checkpoint policy/policy_flat.pt --enable-trajectory  --trajectory-scale 5.0

# To verify droid-slam accuracy or camera placement, you can collect data from wheeled robot as well:
./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_wheeled_robot.py --enable_cameras --udp

# Second terminal
# precise udp command through terminal input
python3 scripts/utils/udp/udp_cmd_client_humanoid.py

# alternatively, command presets as keyboard shortcuts
python3 scripts/utils/udp/udp_cmd_client_humanoid.py --mode preset

# or test with single command
# default 0 speed
python3 scripts/utils/udp/udp_cmd_client_humanoid.py --mode single
# single command
python3 scripts/utils/udp/udp_cmd_client_humanoid.py --mode single --vx 1.0 --vy 0.5 --wz 0.2

# udp control with keyboard shortcut combination
python3 scripts/utils/udp/udp_keyboard_control_humanoid.py
```


#### TLDR: Integrated V-SLAM testing
```bash
# h1 humanoid
source /opt/ros/humble/setup.bash

./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_udp.py --infinite-episode --enable_cameras --udp --checkpoint policy/policy_rough.pt

python3 scripts/utils/udp/udp_cmd_client_humanoid.py

# another terminal for droid-slam
source /opt/ros/humble/setup.bash

# in DROID-SLAM folder, change path accordingly
cd submodules/droid_slam

python3 demo_ros2_realtime.py --rgb_topic=/camera/rgb/image_raw --depth_topic=/camera/depth/image_raw --calib=calib/h1.txt --upsample --asynchronous --publish_pose
# or
python3 demo.py --imagedir=$(vln-humanoids-home)/camera_feed/legged/rgb --depthdir=$(vln-humanoids-home)/camera_feed/legged/depth --calib=calib/h1.txt --stride 1 --upsample



# wheeled robot
source /opt/ros/humble/setup.bash

./submodules/isaaclab/isaaclab.sh -p scripts/isaac_inference_wheeled_robot.py --enable_cameras --udp

python3 scripts/utils/udp/udp_keyboard_control_wheeled.py

# in DROID-SLAM folder, change path accordingly
cd submodules/droid_slam

python3 demo.py --imagedir=$(vln-humanoids-home)/camera_feed/wheeled/rgb --depthdir=$(vln-humanoids-home)/camera_feed/wheeled/depth --calib=calib/wheeled_robot.txt --stride 1 --upsample
```
