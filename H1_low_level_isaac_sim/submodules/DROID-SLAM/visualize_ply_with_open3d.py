#!/usr/bin/env python3
"""
使用Open3D可视化PLY点云文件，并显示世界坐标系
用于对比DROID-SLAM重建的点云与仿真环境
"""

import open3d as o3d
import numpy as np
import os
import glob

def load_and_visualize_ply(ply_file_path):
    """加载PLY文件并用Open3D可视化"""
    
    # 检查文件是否存在
    if not os.path.exists(ply_file_path):
        print(f"文件不存在: {ply_file_path}")
        return
    
    print(f"正在加载点云文件: {ply_file_path}")
    
    # 加载点云
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    if len(pcd.points) == 0:
        print("点云文件为空或格式不正确")
        return
    
    print(f"成功加载点云，包含 {len(pcd.points)} 个点")
    
    # 创建世界坐标系（坐标轴）
    # 坐标轴大小可以根据点云范围调整
    coord_frame_size = 1.0  # 坐标轴长度（米）
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_frame_size, 
        origin=[0, 0, 0]
    )
    
    # 打印点云统计信息
    print(f"点云边界框:")
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"  Min: {bbox.min_bound}")
    print(f"  Max: {bbox.max_bound}")
    print(f"  Center: {bbox.get_center()}")
    
    # 计算点云中心，用于调整视角
    center = pcd.get_center()
    print(f"点云中心: {center}")
    
    # 可选：对点云进行一些预处理
    # 移除离群点
    # pcd, outlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # print(f"移除离群点后剩余 {len(pcd.points)} 个点")
    
    # 基于Z轴高度过滤点云（排除高度大于1.0和小于-0.95的点）
    points = np.asarray(pcd.points)
    # z_filter = (points[:, 2] >= -0.95) & (points[:, 2] <= 1.0)
    
    # # 如果点云有颜色信息，也需要过滤
    # if len(pcd.colors) > 0:
    #     colors = np.asarray(pcd.colors)
    #     filtered_colors = colors[z_filter]
    #     pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # # 如果点云有法向量，也需要过滤
    # if len(pcd.normals) > 0:
    #     normals = np.asarray(pcd.normals)
    #     filtered_normals = normals[z_filter]
    #     pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    
    # # 应用Z轴过滤
    # filtered_points = points[z_filter]
    filtered_points = points
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    print(f"Z轴高度过滤（-0.95 <= z <= 1.0）后剩余 {len(pcd.points)} 个点")

    # 估计法向量（用于更好的可视化效果）
    pcd.estimate_normals()
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="DROID-SLAM Point Cloud Visualization", width=1200, height=800)
    
    # 添加点云和坐标系到场景
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 5.0  # 增大点的尺寸
    
    vis.add_geometry(world_frame)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景

    # 设置视角
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, 1.0])   # 相机朝向
    view_control.set_lookat(center)            # 看向点云中心
    view_control.set_up([0.0, 1.0, 0.0])      # 上方向
    view_control.set_zoom(0.5)
    
    print("\n可视化控制说明:")
    print("- 鼠标左键拖拽: 旋转视角")
    print("- 鼠标右键拖拽: 平移视角") 
    print("- 鼠标滚轮: 缩放")
    print("- 按 'R' 键: 重置视角")
    print("- 按 'ESC' 或关闭窗口: 退出")
    print("\n坐标系说明:")
    print("- 红色轴: X轴")
    print("- 绿色轴: Y轴") 
    print("- 蓝色轴: Z轴")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def find_latest_ply_file(search_dir="."):
    """查找最新的PLY文件"""
    ply_files = glob.glob(os.path.join(search_dir, "accumulated_pointcloud_*.ply"))
    if not ply_files:
        ply_files = glob.glob(os.path.join(search_dir, "*.ply"))
    
    if not ply_files:
        return None
    
    # 按修改时间排序，返回最新的文件
    latest_file = max(ply_files, key=os.path.getmtime)
    return latest_file

def main():
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了文件路径参数
        ply_file = sys.argv[1]
    else:
        # 自动查找最新的PLY文件
        ply_file = find_latest_ply_file()
        if ply_file is None:
            print("未找到PLY文件。请确保:")
            print("1. DROID-SLAM可视化器已运行并保存了点云文件")
            print("2. 或者手动指定PLY文件路径: python visualize_ply_with_open3d.py <ply_file_path>")
            return
        else:
            print(f"自动找到最新的PLY文件: {ply_file}")
    
    load_and_visualize_ply(ply_file)

if __name__ == "__main__":
    main()
