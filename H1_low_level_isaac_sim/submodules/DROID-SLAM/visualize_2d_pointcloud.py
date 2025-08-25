#!/usr/bin/env python3
"""
2D点云可视化脚本
用于显示从DROID-SLAM累积的压平点云数据的2D散点图
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from pathlib import Path
from sklearn.cluster import DBSCAN

def read_ply_file(filepath):
    """读取PLY文件并返回点云数据"""
    points = []
    colors = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # 解析头部信息
        header_end = 0
        vertex_count = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('end_header'):
                header_end = i + 1
                break
        
        # 读取顶点数据
        for i in range(header_end, header_end + vertex_count):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 6:  # x, y, z, r, g, b
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                    
                    points.append([x, y, z])
                    colors.append([r/255.0, g/255.0, b/255.0])
        
        return np.array(points), np.array(colors)
        
    except Exception as e:
        print(f"读取PLY文件失败: {e}")
        return None, None

def plot_2d_pointcloud(points, colors, title="2D Point Cloud", save_path=None, 
                       filter_outliers=True, eps=0.1, min_samples=5):
    """绘制2D散点图，可选择使用DBSCAN过滤离群点
    
    Args:
        points: 点云数据 (N, 3)
        colors: 颜色数据 (N, 3)  
        title: 图表标题
        save_path: 保存路径
        filter_outliers: 是否使用DBSCAN过滤离群点
        eps: DBSCAN的邻域半径参数
        min_samples: DBSCAN的最小样本数参数
    """
    if points is None or len(points) == 0:
        print("没有点云数据可显示")
        return
    
    original_point_count = len(points)
    filtered_points = points.copy()
    filtered_colors = colors.copy()
    
    # 使用DBSCAN过滤离群点
    if filter_outliers and len(points) > min_samples:
        print(f"使用DBSCAN过滤离群点 (eps={eps}, min_samples={min_samples})...")
        
        # 只对xy坐标进行聚类
        xy_points = points[:, :2]
        
        # 应用DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(xy_points)
        
        # 过滤掉离群点（标签为-1的点）
        inlier_mask = cluster_labels != -1
        filtered_points = points[inlier_mask]
        filtered_colors = colors[inlier_mask]
        
        outlier_count = original_point_count - len(filtered_points)
        print(f"原始点数: {original_point_count}, 过滤后点数: {len(filtered_points)}, 离群点数: {outlier_count}")
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制2D散点图（只使用x,y坐标）
    scatter = plt.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                         c=filtered_colors, s=1, alpha=0.6, edgecolors='none')
    
    plt.title(title, fontsize=14)
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # 保持x,y轴比例相同
    
    # 添加统计信息
    x_range = np.max(filtered_points[:, 0]) - np.min(filtered_points[:, 0])
    y_range = np.max(filtered_points[:, 1]) - np.min(filtered_points[:, 1])
    info_text = f"Points: {len(filtered_points)}"
    if filter_outliers:
        info_text += f" (原始: {original_point_count})"
    info_text += f"\nX Range: {x_range:.2f}m\nY Range: {y_range:.2f}m"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D散点图已保存到: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='可视化DROID-SLAM的2D点云数据')
    parser.add_argument('--file', '-f', type=str, required=True, help='指定要可视化的PLY文件路径')
    
    args = parser.parse_args()
    
    # 可视化指定文件
    print(f"可视化指定文件: {args.file}")
    points, colors = read_ply_file(args.file)
    
    if points is not None:
        # 如果颜色数据为空，创建默认颜色
        if colors is None or len(colors) == 0:
            colors = np.zeros((len(points), 3))
            
        filename = os.path.basename(args.file)
        plot_2d_pointcloud(points, colors, 
                          title=f"2D Point Cloud - {filename}",
                          save_path=f"2d_pointcloud_{filename.replace('.ply', '.png')}",
                          filter_outliers=False,
                          eps=0.1,
                          min_samples=5)
    else:
        print("读取点云数据失败")

if __name__ == "__main__":
    main()
