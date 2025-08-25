#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import open3d as o3d

# intrinsics
WIDTH  = 640
HEIGHT = 480
FOCAL_LENGTH_MM       = 24.0       # CameraCfg.spawn.focal_length
HORIZONTAL_APERTUREMM = 20.955     # CameraCfg.spawn.horizontal_aperture

CX = WIDTH / 2.0
CY = HEIGHT / 2.0


RGB_PATH   = "camera_feed/rgb.png"
DEPTH_PATH = "camera_feed/depth.npy"
OUT_PLY    = "camera_feed/cloud_world.ply"
DEPTH_MIN  = 0.05
DEPTH_MAX  = 200.0
STRIDE     = 1


def usd_to_fx_fy(width, height, focal_mm, horiz_ap_mm):
    """
    USD/Omniverse 相机换算:
      fx = focal_length(mm) / horizontalAperture(mm) * width(px)
      fy = focal_length(mm) / verticalAperture(mm)   * height(px)
    其中 verticalAperture = horizontalAperture * (height/width)
    => fy 与 fx 数值相同（方形像素时）。
    """
    fx = focal_mm * width  / horiz_ap_mm
    # vertical_aperture
    vert_ap_mm = horiz_ap_mm * (height / width)
    fy = focal_mm * height / vert_ap_mm
    return fx, fy


def make_pointcloud(rgb_u8, depth_m, fx, fy, cx, cy, stride=1,
                    dmin=0.05, dmax=50.0):

    h, w = depth_m.shape
    assert rgb_u8.shape[0] == h and rgb_u8.shape[1] == w

    # sample points
    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    us = us.astype(np.float32)
    vs = vs.astype(np.float32)

    z = depth_m[vs.astype(np.int32), us.astype(np.int32)]
    valid = np.isfinite(z) & (z > dmin) & (z < dmax)

    us = us[valid]; vs = vs[valid]; z = z[valid]

    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    # camera coordinate system: x right, y down, z forward
    pts = np.stack([x, y, z], axis=-1)

    colors = rgb_u8[vs.astype(np.int32), us.astype(np.int32), :].astype(np.float32) / 255.0
    return pts, colors


def main():
    rgb_bgr = cv2.imread(RGB_PATH, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {RGB_PATH}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth = np.load(DEPTH_PATH).astype(np.float32)

    if rgb.shape[0] != HEIGHT or rgb.shape[1] != WIDTH:
        raise ValueError(f"RGB size {rgb.shape[:2]} != expected {(HEIGHT, WIDTH)}")
    if depth.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f"Depth size {depth.shape[:2]} != expected {(HEIGHT, WIDTH)}")

    # compute intrinsics
    fx, fy = usd_to_fx_fy(WIDTH, HEIGHT, FOCAL_LENGTH_MM, HORIZONTAL_APERTUREMM)
    print(f"[K] fx={fx:.3f}  fy={fy:.3f}  cx={CX:.3f}  cy={CY:.3f}")

    # 反投影
    pts, colors = make_pointcloud(rgb, depth, fx, fy, CX, CY, stride=STRIDE,
                                dmin=DEPTH_MIN, dmax=DEPTH_MAX)

    print(f"[Stats] points={pts.shape[0]}")
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size:
        print(f"[Depth] min/median/max = {finite.min():.3f} / {np.median(finite):.3f} / {finite.max():.3f} m")

    # Open3D visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(OUT_PLY) or ".", exist_ok=True)
    o3d.io.write_point_cloud(OUT_PLY, pcd, write_ascii=True)
    print(f"[OK] saved PLY -> {OUT_PLY}")

    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)],
        window_name="Single Camera PointCloud",
    )


if __name__ == "__main__":
    main()
