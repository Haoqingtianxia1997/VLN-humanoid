import os
import numpy as np
import torch    


# TODO: NOT properly tuned yet! I intended to tune for roation only scenario, because difference between target vel and actual vel of vx and vy 
# can be fixed with a simple least squares fit. So I wanted to have a higer p gain for faster response. but when h1 is moving along x/y axis, the 
# random rotation error will cause the controller to oscillate a bit. 
Kp = 1.0
Ki = 0.0
Kd = 0.5

err_last_wz = 0.0
integral_wz = 0.0

def pid_control_wz(target, actual, dt, output_limit=(-1.0, 1.0), integral_limit=0.5):
    global err_last_wz, integral_wz

    err = actual - target # rot direction wrong for some reason, so we use actual - target
    derr = (err - err_last_wz) / dt if dt > 0 else 0.0

    integral_wz += err * dt
    # anti-windup clip
    integral_wz = max(min(integral_wz, integral_limit), -integral_limit)

    output = Kp * err  + Ki * integral_wz + Kd * derr  
    # clip output
    output = max(min(output, output_limit[1]), output_limit[0])

    err_last_wz = err
    return output


def correct_velocity_command(target_vx, target_vy, target_wz, obs_monitor, args_cli, env):
    """ Corrects the velocity command based on observed velocities and PID control.
    Args:
        target_vx (float): Target forward velocity.
        target_vy (float): Target lateral velocity.
        target_wz (float): Target angular velocity.
        obs_monitor (ObservationMonitor): Monitor for observed velocities.
        args_cli: Command line arguments containing device information.
    Returns:
        torch.Tensor: Corrected velocity command tensor.
    """  
    # least squares fit for open loop linear velocity control
    vx_corrected = 1.0541 * target_vx - 0.0257
    vy_corrected = 1.0382 * target_vy - 0.0097

    # least squares fit for angular velocity
    wz_ls = -0.4524 * target_wz - 0.0662
    actual_wz = obs_monitor.avg_velocity[2] if hasattr(obs_monitor, 'avg_velocity') else 0.0
    wz_corrected = pid_control_wz(target_wz, actual_wz, env.step_dt) + wz_ls
    
    cmd_tensor = torch.tensor([vx_corrected, vy_corrected, 0.0, target_wz], device=args_cli.device)

    return cmd_tensor