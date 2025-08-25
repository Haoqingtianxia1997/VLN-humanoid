import subprocess
import json
import time
from src.VLM_agent.agent import VLM_agent  
import socket, struct, time
import numpy as np
import cv2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1", 5555)

def call_ros2_service(service_name, service_type, args_dict):
    """
    调用 ROS 2 服务，通过 subprocess 调用 CLI，等待结果并返回是否成功。
    """
    arg_str = json.dumps(args_dict).replace('"', '\\"')
    cmd = [
        "ros2", "service", "call",
        service_name,
        service_type,
        arg_str
    ]
    print(f"\n🚀 Calling service: {' '.join(cmd)}")

    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        print("✅ Service call returned:")
        print(result)

        # 检查是否包含 success: True
        if "success: true" in result.lower():
            return True
        else:
            print("❌ Service reported failure.")
            return False
    except subprocess.CalledProcessError as e:
        print("❌ Service call failed:")
        print(e.output)
        return False


def send(vx, vy, wz):
    """把三个 float32 发送给仿真端"""
    sock.sendto(struct.pack("fff", vx, vy, wz), addr)
    
    
def execute_action_sequence(actions):
    """
    串行执行动作序列，每一步等待其服务执行完且成功后才进行下一步。
    """
    target_point = None
    angle = 0  # 初始化角度变量，后续可根据需要修改
    distance = 0  # 初始化距离变量，后续可根据需要修改
    direction = None  # 初始化方向变量，后续可根据需要修改
    
    for i, action in enumerate(actions):
        print(f"\n▶️ Executing action {i+1}/{len(actions)}: {action}")
        act_type = action["type"]
        if action["target"] is not None:
            target = action["target"]
        params = action.get("parameters", {})
        if "distance" in params:
            distance = params["distance"]
        if "angle" in params:
            angle = params["angle"]
        if "direction" in params:
            direction = params["direction"]

        if act_type == "perceive":
            pass
            success = True
                       
        elif act_type == "planning":
            print(f"execute planning action, plan to move to '{target}', target location:", target_point)
            time.sleep(5)
            success = True
            # success = call_ros2_service("/robot/planning", "your_msgs/srv/Planning", {"target": target})
        elif act_type == "decision_making":
            print("execute decision_making action")
            time.sleep(5)
            success = True
            # success = call_ros2_service("/robot/decision_making", "your_msgs/srv/Decision", {"target": target})
        elif act_type == "execution":
            print("execute execution action")
            time.sleep(5)
            success = True
            # success = call_ros2_service("/robot/execution", "your_msgs/srv/Sxecution", {"target": target})
        elif act_type == "move_forward":
            print("execute move_Forward action, distance:", distance)
            t =  distance / 0.50  # 假设速度为 0.50 m/s
            send(0.50, 0.00, 0.00)
            time.sleep(t)
            send(0.04, 0.00, 0.00)
            time.sleep(1)  # 模拟执行时间
            success = True
            # success = call_ros2_service("/robot/forward", "your_msgs/srv/Forward", {"parameters": { "distance": distance })
        elif act_type == "move_backward":
            print("execute move_backward action, distance:", distance)
            t =  distance / 0.50  # 假设速度为 0.50 m/s
            send(-0.50, 0.00, 0.00)
            time.sleep(t)
            send(0.04, 0.00, 0.00)
            time.sleep(1)  # 模拟执行时间
            success = True
            # success = call_ros2_service("/robot/backward", "your_msgs/srv/Backward", {"parameters": { "distance": distance })
        elif act_type == "move_left":
            print("execute move_left action, distance:", distance)
            t =  distance / 0.50  # 假设速度为 0.50 m/s
            send(0.00, 0.5, 0.00)
            time.sleep(t)
            send(0.04, 0.00, 0.00)
            time.sleep(1)  # 模拟执行时间
            success = True
            # success = call_ros2_service("/robot/left", "your_msgs/srv/Left", {"parameters": { "distance": distance })            
        elif act_type == "move_right":
            print("execute move_right action, distance:", distance) 
            t =  distance / 0.50  # 假设速度为 0.50 m/s
            send(0.00, -0.5, 0.00)
            time.sleep(t)
            send(0.04, 0.00, 0.00)
            time.sleep(1)  # 模拟执行时间
            success = True
            # success = call_ros2_service("/robot/right", "your_msgs/srv/Right", {"parameters": { "distance": distance })
        elif act_type == "turn":
            print("execute turning action, angle:", angle, "direction:", direction)
            angle = angle / 180 * 3.14  # 将角度转换为弧度
            t =  angle / 0.20  # 假设速度为 0.50 m/s
            wy = 0.2 * (1 if direction == "left" else -1) 
            send(0.00, 0.0, wy)
            time.sleep(t)
            send(0.04, 0.00, 0.00)
            time.sleep(1)  # 模拟执行时间
            success = True
            # success = call_ros2_service("/robot/right", "your_msgs/srv/Right", {"parameters": { "angle": angle, "direction": direction }})
        else:
            print(f"⚠️ Unknown action type: {act_type}")
            success = False

        # 检查服务是否成功
        if not success:
            print(f"⛔ Aborting action sequence due to failure at step {i+1}.")
            break

        time.sleep(0.5)  # 可选延迟
    
    print("✅ Action sequence completed.")

# ✅ 测试用例：手动构造动作序列
if __name__ == "__main__":
    actions = [
        {"type": "perceive", "target": "cup"},
        {"type": "move", "target": "apple"},
        {"type": "grasp", "target": "apple"},
        {"type": "move", "target": "plate"},
        {"type": "place", "target": "plate"},
        {"type": "reset", "target": "home"}
    ]

    execute_action_sequence(actions)



# # 举例 your_msgs/srv/Move.srv
# string target
# ---
# bool success
# string message

# 服务端必须返回：
# return Move.Response(success=True, message="Moved to apple.")
