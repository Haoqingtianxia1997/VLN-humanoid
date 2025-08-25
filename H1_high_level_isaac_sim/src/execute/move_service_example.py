#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from your_msgs.srv import Move
from sensor_msgs.msg import JointState
from your_custom_msgs.msg import LowCmd  # 替换为你实际消息类型
import numpy as np
import time

class MoveService(Node):
    def __init__(self):
        super().__init__('move_service')
        self.srv = self.create_service(Move, '/robot/move', self.move_callback)
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)
        self.state_sub = self.create_subscription(JointState, '/lowstate', self.state_callback, 10)
        self.current_joints = None
        self.get_logger().info("✅ /robot/move service is ready.")

    def state_callback(self, msg):
        self.current_joints = list(msg.position)

    def inverse_kinematics(self, world_position) -> list:
        """
        将世界坐标解算为目标关节角（伪代码，需你替换成实际 IK 计算）
        """
        # 示例：你可以调用 MoveIt、KDL、自己的 DH 模型等
        # 举例：return ik_solver.solve(position=world_position)
        raise NotImplementedError("❌ inverse_kinematics() 需要你接入 IK 求解器")

    def interpolate_joints(self, start, goal, steps) -> list:
        """
        在线性空间中插值关节路径（返回 list of joint positions）
        """
        start = np.array(start)
        goal = np.array(goal)
        return [list(start + (goal - start) * (i / steps)) for i in range(1, steps + 1)]

    def publish_joint_command(self, joint_angles):
        msg = LowCmd()
        msg.joint_position = joint_angles  # 替换成你真实的字段
        self.cmd_pub.publish(msg)

    def move_callback(self, request, response):
        world_target = request.target  # 你应定义为 geometry_msgs/Point 或 Pose
        if self.current_joints is None:
            response.success = False
            response.message = "❌ 当前状态未知，无法执行移动"
            return response

        try:
            goal_joints = self.inverse_kinematics(world_target)
        except Exception as e:
            response.success = False
            response.message = f"❌ IK 求解失败: {e}"
            return response

        current = self.current_joints
        steps = 50
        tolerance = 0.01
        duration = 5.0
        rate_hz = steps / duration
        rate = self.create_rate(rate_hz)

        path = self.interpolate_joints(current, goal_joints, steps)
        self.get_logger().info(f"🦾 执行插值移动，共 {steps} 步")

        for idx, joint_pos in enumerate(path):
            self.publish_joint_command(joint_pos)
            rate.sleep()

        # 最终等待到位确认
        timeout = self.get_clock().now().seconds_nanoseconds()[0] + 3
        while rclpy.ok():
            if self.current_joints is None:
                continue
            error = np.abs(np.array(goal_joints) - np.array(self.current_joints))
            if np.all(error < tolerance):
                response.success = True
                response.message = "✅ 已成功到达目标"
                return response
            if self.get_clock().now().seconds_nanoseconds()[0] > timeout:
                response.success = False
                response.message = "⚠️ 到位检测超时"
                return response
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = MoveService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
