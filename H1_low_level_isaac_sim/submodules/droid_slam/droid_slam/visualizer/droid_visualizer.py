import moderngl
import numpy as np
from lietorch import SE3

import torch
import droid_backends
import moderngl_window
import moderngl
from moderngl_window.opengl.vao import VAO

import numpy as np
from .camera import OrbitDragCameraWindow
from align import align_pose_fragements

# 添加 matplotlib 用于 3D 轨迹可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading

# 添加UDP传输相关模块
import socket
import json
import time

CAM_POINTS = 0.05 * np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
).astype("f4")

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)


CAM_SEGMENTS = []
for i, j in CAM_LINES:
    CAM_SEGMENTS.append(CAM_POINTS[i])
    CAM_SEGMENTS.append(CAM_POINTS[j])

CAM_SEGMENTS = np.stack(CAM_SEGMENTS, axis=0)


def quaternion_to_rotation_matrix(w, x, y, z):
    """将四元数(w,x,y,z)转换为旋转矩阵"""
    # 归一化四元数
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 构建旋转矩阵
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])
    return R


def quaternion_multiply(q1, q2):
    """四元数乘法 q1 * q2，四元数格式为[w,x,y,z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def merge_depths_and_poses(depth_video1, depth_video2):
    t1 = depth_video1.counter.value
    t2 = depth_video2.counter.value

    poses1 = depth_video1.poses[:max(t1, t2)].clone()
    poses2 = depth_video2.poses[:max(t1, t2)].clone()

    disps1 = depth_video1.disps[:max(t1, t2)].clone()
    disps2 = depth_video2.disps[:max(t1, t2)].clone()

    if t2 <= 0:
        return poses1, disps1
    
    if t2 >= t1:
        return poses2, disps2
    
    dP, s = align_pose_fragements(
        poses1[max(0, t2-16): t2],
        poses2[max(0, t2-16): t2],
    )

    poses1[..., :3] *= s

    poses2[t2:] = (dP * SE3(poses1[t2:])).data
    disps2[t2:] = disps1[t2:] / s

    return poses2, disps2


class DroidVisualizer(OrbitDragCameraWindow):
    title = "Droid Visualizer"
    _depth_video1 = None
    _depth_video2 = None

    _refresh_rate = 5
    _filter_threshold = 0.02
    _filter_count = 2
    
    # 添加类变量来存储轨迹数据
    _latest_trajectory_data = None
    
    # UDP相关配置
    _udp_socket = None
    _udp_host = "127.0.0.1"
    _udp_port = 12346
    
    # 角度数据UDP接收相关
    _angle_udp_socket = None
    _angle_udp_port = 12347
    _latest_angle_data = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = False
        
        # 初始化UDP socket
        self._init_udp_socket()
        
        # 初始化角度UDP接收器
        self._init_angle_udp_receiver()

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;
                in vec3 in_color0;
                in float in_alpha0;

                uniform mat4 m_proj;
                uniform mat4 m_cam;

                out vec3 color;
                out float alpha;

                void main() {
                    gl_Position = m_proj * m_cam * vec4(in_position, 1.0);
                    color = in_color0;
                    alpha = in_alpha0;
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                in vec3 color;
                in float alpha;

                void main()
                {
                    if (alpha <= 0)
                        discard;

                    fragColor = vec4(color, alpha);
                }
            """,
        )

        self.cam_prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;

                uniform mat4 m_proj;
                uniform mat4 m_cam;

                void main() {
                    gl_Position = m_proj * m_cam * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                out vec4 fragColor;
                uniform vec3 color;

                void main()
                {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        n, h, w = self._depth_video1.disps.shape
        max_num_points = n * h * w

        # Upload buffer to GPU
        valid = np.zeros((max_num_points,), dtype="f4")
        points = np.zeros((max_num_points, 3), dtype="f4")
        colors = np.zeros((max_num_points, 3), dtype="f4")

        self.valid_buffer = self.ctx.buffer(valid.tobytes())
        self.pts_buffer = self.ctx.buffer(points.tobytes())
        self.clr_buffer = self.ctx.buffer(colors.tobytes())

        self.vao = VAO("geometry_frustrum", mode=moderngl.LINES)
        self.cam_prog["color"].value = (0, 0, 0)

        # cam_segments = CAM_SEGMENTS.repeat(n, 1).astype(np.float32)
        # print(cam_segments.shape)
        cam_segments = CAM_SEGMENTS.astype("f4")
        cam_segments = np.tile(cam_segments, (n, 1))


        self.count = 0

        # Create a vertex array manually
        self.points = self.ctx.vertex_array(
            self.prog,
            [
                (self.pts_buffer, "3f", "in_position"),
                (self.clr_buffer, "3f", "in_color0"),
                (self.valid_buffer, "f", "in_alpha0"),
            ],
        )

        self.cam_buffer = self.ctx.buffer(cam_segments.tobytes())
        self.cams = self.ctx.vertex_array(
            self.cam_prog,
            [
                (self.cam_buffer, "3f", "in_position"),
            ],
        )

        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom = 1.0

        # 初始化 3D 轨迹可视化
        self._init_3d_trajectory_plot()

    def _init_udp_socket(self):
        """初始化UDP socket用于发送轨迹数据"""
        try:
            DroidVisualizer._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        except Exception as e:

            DroidVisualizer._udp_socket = None
    
    def _init_angle_udp_receiver(self):
        """初始化角度UDP接收器"""
        try:
            DroidVisualizer._angle_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            DroidVisualizer._angle_udp_socket.bind((DroidVisualizer._udp_host, DroidVisualizer._angle_udp_port))
            DroidVisualizer._angle_udp_socket.settimeout(0.001)  # 非阻塞接收
            
            # 启动角度数据接收线程
            import threading
            self.angle_thread = threading.Thread(target=self._angle_receiver_thread, daemon=True)
            self.angle_thread.start()
            print(f"Angle UDP receiver started on port {DroidVisualizer._angle_udp_port}")
            
        except Exception as e:
            print(f"Failed to initialize angle UDP receiver: {e}")
            DroidVisualizer._angle_udp_socket = None
    
    def _angle_receiver_thread(self):
        """四元数数据接收线程"""
        while True:
            try:
                if DroidVisualizer._angle_udp_socket:
                    data, addr = DroidVisualizer._angle_udp_socket.recvfrom(1024)
                    json_data = data.decode('utf-8')
                    quaternion_data = json.loads(json_data)
                    DroidVisualizer._latest_angle_data = quaternion_data
                    # print(f"Received quaternion data: {quaternion_data}")
            except socket.timeout:
                pass
            except Exception as e:
                # print(f"Error in angle receiver thread: {e}")
                pass
            time.sleep(0.001)

    def _send_trajectory_data_udp(self, trajectory_data):
        """通过UDP发送轨迹数据"""
        if DroidVisualizer._udp_socket is None:
            return
            
        try:
            # 将数据序列化为JSON
            json_data = json.dumps(trajectory_data)
            data_bytes = json_data.encode('utf-8')
            
            # 发送数据
            DroidVisualizer._udp_socket.sendto(data_bytes, (DroidVisualizer._udp_host, DroidVisualizer._udp_port))

        except Exception as e:
            print(f"Failed to send trajectory data via UDP: {e}")

    def _init_3d_trajectory_plot(self):
        """初始化 3D 轨迹可视化"""
        # 设置 matplotlib 非阻塞模式
        plt.ion()
        
        # 创建 3D 图
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('DROID-SLAM real time trajectory(3D)')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 初始化轨迹线和点
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.current_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Current Position')

        # 设置坐标轴等比例
        self.ax.set_box_aspect([1,1,1])
        self.ax.legend()
        
        # 存储轨迹数据
        self.trajectory_points = []
        
        plt.show(block=False)

    def _update_3d_trajectory_from_cam_pts(self, cam_pts, frame_count):
        """从相机变换后的点云数据更新 3D 轨迹图"""
        try:
            if cam_pts.shape[0] > 0:
                # cam_pts 是 [frame_count * 20, 3] 的形状
                # 其中 frame_count 是帧数，20 是每个相机框架的线段端点数
                points_per_frame = 20  # CAM_SEGMENTS 有20个点（10条线段×2个端点）
                
                if cam_pts.shape[0] >= points_per_frame * frame_count:
                    # 重新组织数据为 [frame_count, points_per_frame, 3]
                    cam_pts_reshaped = cam_pts[:frame_count * points_per_frame].reshape(frame_count, points_per_frame, 3)
                    
                    # 计算每帧相机的中心点作为轨迹点
                    camera_positions = []
                    for frame_idx in range(frame_count):
                        # 计算当前帧所有20个点的平均位置作为相机中心
                        camera_center_slam = np.mean(cam_pts_reshaped[frame_idx, :, :], axis=0)  # SLAM坐标系中的相机位置
                        
        
                        if DroidVisualizer._latest_angle_data is not None:
                            torso_quat_data = DroidVisualizer._latest_angle_data.get('torso_to_world_quat', {})
                            torso_to_world_quat = np.array([
                                torso_quat_data.get('w', 1.0),
                                torso_quat_data.get('x', 0.0),
                                torso_quat_data.get('y', 0.0),
                                torso_quat_data.get('z', 0.0)
                            ])
                        
                        # 相机到torso的变换（根据您提供的四元数）
                        camera_to_torso_quat = np.array([0.45451948, -0.54167522, 0.54167522, -0.45451948])
                        
                        # 计算相机到世界坐标系的四元数：camera_to_world = torso_to_world * camera_to_torso
                        camera_to_world_quat = quaternion_multiply(torso_to_world_quat, camera_to_torso_quat)
                        # 将四元数转换为旋转矩阵
                        R_camera_to_world = quaternion_to_rotation_matrix(
                            camera_to_world_quat[0], camera_to_world_quat[1], 
                            camera_to_world_quat[2], camera_to_world_quat[3]
                        )
                        
                        # 关键修改：camera_center_slam是SLAM相机坐标系中的位置，需要正确变换
                        # 将SLAM相机坐标系中的相机位置变换到世界坐标系
                        camera_position_world = R_camera_to_world @ camera_center_slam
                        
                        camera_positions.append(camera_position_world)
                    
                    camera_positions = np.array(camera_positions)
                    
                    # 更新轨迹数据
                    self.trajectory_points = camera_positions.tolist()
                    
                    # 存储最新的轨迹数据到类变量（供外部访问）
                    if len(self.trajectory_points) > 0:
                        # 提取x和z坐标（3D plot中的x对应真实x，z对应真实y）
                        x_coords = [p[0] for p in self.trajectory_points]
                        y_coords = [p[1] for p in self.trajectory_points]
                        z_coords = [p[2] for p in self.trajectory_points]
                        trajectory_data = {
                            'x_coords': x_coords,
                            'y_coords': y_coords,
                            'z_coords': z_coords,
                            'frame_count': frame_count,
                            'latest_position': {
                                'x': x_coords[-1],     
                                'y': y_coords[-1],      
                                'z': z_coords[-1]      
                            },
                            'timestamp': time.time()
                        }
                        
                        DroidVisualizer._latest_trajectory_data = trajectory_data
                        
                        # 通过UDP发送轨迹数据
                        self._send_trajectory_data_udp(trajectory_data)
                    
                    # 更新轨迹线
                    if len(self.trajectory_points) > 1:
                        x_data = [p[0] for p in self.trajectory_points]
                        y_data = [p[1] for p in self.trajectory_points]
                        z_data = [p[2] for p in self.trajectory_points]
                        
                        self.trajectory_line.set_data_3d(x_data, y_data, z_data)
                        
                        # 更新当前位置点
                        self.current_point.set_data_3d([x_data[-1]], [y_data[-1]], [z_data[-1]])
                        
                        # 动态调整坐标轴范围，保持各轴分度值相同
                        margin = 0.5  # 边距
                        
                        # 计算每个轴的数据范围
                        x_min, x_max = min(x_data) - margin, max(x_data) + margin
                        y_min, y_max = min(y_data) - margin, max(y_data) + margin
                        z_min, z_max = min(z_data) - margin, max(z_data) + margin
                        
                        # 找到最大的数据范围
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        z_range = z_max - z_min
                        max_range = max(x_range, y_range, z_range)
                        
                        # 计算每个轴的中心点
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        z_center = (z_min + z_max) / 2
                        
                        # 使用最大范围设置所有轴的范围，保持相同的分度值
                        half_range = max_range / 2
                        self.ax.set_xlim(x_center - half_range, x_center + half_range)
                        self.ax.set_ylim(y_center - half_range, y_center + half_range)
                        self.ax.set_zlim(z_center - half_range, z_center + half_range)
                    
                    # 刷新图形
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    
        except Exception as e:
            print(f"Error when updating 3D trajectory from cam_pts: {e}")

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.ctx.point_size = 0.2

        self.prog["m_proj"].write(self.camera.projection.matrix)
        self.prog["m_cam"].write(self.camera.matrix)

        self.cam_prog["m_proj"].write(self.camera.projection.matrix)
        self.cam_prog["m_cam"].write(self.camera.matrix)

        t = self._depth_video1.counter.value

        if t > 12 and self.count % self._refresh_rate == 0:
            images = self._depth_video1.images[:t, :, 4::8, 4::8]
            intrinsics = self._depth_video1.intrinsics

            if self._depth_video2 is not None:
                poses, disps = merge_depths_and_poses(self._depth_video1, self._depth_video2)
                poses = poses[:t]
                disps = disps[:t]
            else:
                disps = self._depth_video1.disps[:t]
                poses = self._depth_video1.poses[:t]

            # 4x4 homogenous matrix
            cam_pts = torch.from_numpy(CAM_SEGMENTS).cuda()
            cam_pts = SE3(poses[:, None]).inv() * cam_pts[None]
            cam_pts = cam_pts.reshape(-1, 3).cpu().numpy()

            self._update_3d_trajectory_from_cam_pts(cam_pts, t)

            self.cam_buffer.write(cam_pts)

            index = torch.arange(t, device="cuda")
            thresh = self._filter_threshold * torch.ones_like(disps.mean(dim=[1, 2]))

            points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
            colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

            counts = droid_backends.depth_filter(
                poses, disps, intrinsics[0], index, thresh
            )
            
            # 确保所有张量的尺寸匹配
            min_size = min(counts.shape[0], disps.shape[0], points.shape[0], colors.shape[0])
            counts = counts[:min_size]
            disps_for_mask = disps[:min_size]
            points = points[:min_size]
            colors = colors[:min_size]
            
            mask = (counts >= self._filter_count) & (disps_for_mask > 0.25 * disps_for_mask.mean())

            valid = mask.float()

            # wasteful (gpu -> cpu -> gpu)
            self.pts_buffer.write(points.contiguous().cpu().numpy())
            self.clr_buffer.write(colors.contiguous().cpu().numpy())
            self.valid_buffer.write(valid.contiguous().cpu().numpy())

        self.count += 1
        self.points.render(mode=moderngl.POINTS)
        self.cams.render(mode=moderngl.LINES)


def visualization_fn(depth_video1, depth_video2):
    config = DroidVisualizer
    config._depth_video1 = depth_video1
    config._depth_video2 = depth_video2

    try:
        # run visualizer
        moderngl_window.run_window_config(config, args=["-r", "True"])
    finally:
        # 清理UDP socket
        if DroidVisualizer._udp_socket:
            try:
                DroidVisualizer._udp_socket.close()
                print("UDP socket closed")
            except Exception as e:
                print(f"Error closing UDP socket: {e}")
        
        # 清理角度UDP socket
        if DroidVisualizer._angle_udp_socket:
            try:
                DroidVisualizer._angle_udp_socket.close()
                print("Angle UDP socket closed")
            except Exception as e:
                print(f"Error closing angle UDP socket: {e}")

def create_udp_receiver(host="127.0.0.1", port=12345, timeout=1.0):
    """创建UDP接收器用于接收轨迹数据"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.settimeout(timeout)
        print(f"UDP receiver created and listening on {host}:{port}")
        return sock
    except Exception as e:
        print(f"Failed to create UDP receiver: {e}")
        return None

def receive_trajectory_data_udp(sock):
    """从UDP接收轨迹数据"""
    try:
        data, addr = sock.recvfrom(65536)  # 64KB buffer
        json_data = data.decode('utf-8')
        trajectory_data = json.loads(json_data)
        return trajectory_data
    except socket.timeout:
        return None
    except Exception as e:
        print(f"Failed to receive trajectory data via UDP: {e}")
        return None
