#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/utils.py
Project: /workspace/code/sam3d_body
Created Date: Thursday December 4th 2025
Author: Kaixu Chen
-----
Comment:
Visualization utilities for SAM3Dbody results.

Have a good code time :)
-----
Last Modified: Thursday December 4th 2025 4:24:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import json
import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np


import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


from .sam_3d_body.visualization.renderer import Renderer
from .sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from .tools.vis_utils import visualize_sample_together

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        # Draw bounding box
        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2,
        )

        # Add person ID text
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    results = []

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # 1. Original image
        img_orig = img_cv2.copy()

        # 2. Mesh overlay on original image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 3. Mesh on white background (front view)
        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 4. Side view
        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        # Combine all views
        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def visualize_3d_skeleton(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    visualizer: SkeletonVisualizer,
) -> np.ndarray:
    """
    3D 骨架现场绘制接口。
    """
    # 1. 初始化 Matplotlib 3D 画布
    # 使用 Agg 后端防止在服务器报错（如果在 main 开头设置过则此处不需要）
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # 设置基础外观
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 2. 预准备颜色数据 (将 0-255 归一化到 0-1)
    kpt_colors = (
        np.array(visualizer.kpt_color, dtype=np.float32) / 255.0
        if visualizer.kpt_color is not None
        else None
    )
    link_colors = (
        np.array(visualizer.link_color, dtype=np.float32) / 255.0
        if visualizer.link_color is not None
        else None
    )

    has_data = False

    # 获取所有人的坐标以统一缩放比例（防止每个人比例不一致）
    all_points = []
    for target in outputs:
        pts = target.get("pred_keypoints_3d")
        if pts is not None:
            all_points.append(pts.reshape(-1, 3))

    if all_points:
        has_data = True
        all_points_np = np.concatenate(all_points, axis=0)

        # 自动调整坐标轴比例，确保人体不变形
        max_range = (all_points_np.max(axis=0) - all_points_np.min(axis=0)).max() / 2.0
        mid = (all_points_np.max(axis=0) + all_points_np.min(axis=0)) / 2.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # 3. 现场开始绘制
        for i, target in enumerate(outputs):
            pts_3d = target.get("pred_keypoints_3d")
            if pts_3d is None:
                continue
            if pts_3d.ndim == 3:
                pts_3d = pts_3d[0]  # 处理 (1, N, 3)

            # 绘制关键点
            ax.scatter(
                pts_3d[:, 0],
                pts_3d[:, 1],
                pts_3d[:, 2],
                c=kpt_colors if kpt_colors is not None else "r",
                s=visualizer.radius * 5,
                alpha=getattr(visualizer, "alpha", 0.8),
            )

            # 绘制骨架连线
            if visualizer.skeleton is not None:
                for j, (p1_idx, p2_idx) in enumerate(visualizer.skeleton):
                    if p1_idx < len(pts_3d) and p2_idx < len(pts_3d):
                        p1, p2 = pts_3d[p1_idx], pts_3d[p2_idx]
                        color = (
                            link_colors[j % len(link_colors)]
                            if link_colors is not None
                            else "b"
                        )
                        ax.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            [p1[2], p2[2]],
                            color=color,
                            linewidth=visualizer.line_width,
                            alpha=getattr(visualizer, "alpha", 0.8),
                        )

    if not has_data:
        ax.text(0.5, 0.5, 0.5, "No Data", ha="center")

    # 设置初始视角 (根据你的经验：俯视角度)
    ax.view_init(elev=-30, azim=270)

    # 4. 转换为图像数组
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_3d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_3d = img_3d.reshape((h, w, 4))[:, :, :3]  # 去掉 alpha 通道

    # 4. 关闭 fig 释放内存
    plt.close(fig)
    return img_3d


def vis_results(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
    visualizer: SkeletonVisualizer,
    cfg: Optional[Dict[str, Any]] = None,
):
    """Save 3D mesh results to files and return PLY file paths"""

    os.makedirs(save_dir, exist_ok=True)

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Save focal length
    if outputs and cfg.get("save_focal_length", False):
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        logger.info(f"Saved focal length: {focal_length_path}")

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # Store individual mesh
        if cfg.get("save_mesh_ply", False):
            tmesh = renderer.vertices_to_trimesh(
                person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
            )
            mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
            mesh_path = os.path.join(save_dir, mesh_filename)
            tmesh.export(mesh_path)

            logger.info(f"Saved mesh ply file: {mesh_path}")

        # Save individual overlay image
        if cfg.get("save_mesh_overlay", False):
            img_mesh_overlay = (
                renderer(
                    person_output["pred_vertices"],
                    person_output["pred_cam_t"],
                    img_cv2.copy(),
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
                * 255
            ).astype(np.uint8)

            overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
            cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)
            logger.info(f"Saved overlay: {os.path.join(save_dir, overlay_filename)}")

        # Save bbox image
        if cfg.get("save_bbox_image", False):
            img_bbox = img_cv2.copy()
            bbox = person_output["bbox"]
            img_bbox = cv2.rectangle(
                img_bbox,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                4,
            )
            bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
            cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)
            logger.info(f"Saved bbox: {os.path.join(save_dir, bbox_filename)}")

        # 2D 结果可视化
        if cfg.get("plot_2d", False):
            vis_results = visualize_2d_results(img_cv2, outputs, visualizer)
            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_2d_visualization.png"),
                vis_results[pid],
            )
            logger.info(
                f"Saved 2D visualization: {os.path.join(save_dir, f'{image_name}_2d_visualization.png')}"
            )

        # 3D 网格可视化
        if cfg.get("save_3d_mesh", False):
            mesh_results = visualize_3d_mesh(img_cv2, outputs, faces)
            # Display results

            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_3d_mesh_visualization_{pid}.png"),
                mesh_results[pid],
            )

            logger.info(
                f"Saved 3D mesh visualization: {os.path.join(save_dir, f'{image_name}_3d_mesh_visualization_{pid}.png')}"
            )

        # 3D kpt可视化
        if cfg.get("save_3d_keypoints", False):
            kpt3d_img = visualize_3d_skeleton(
                img_cv2=img_cv2.copy(), outputs=outputs, visualizer=visualizer
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_3d_kpt_visualization_{pid}.png"),
                kpt3d_img,
            )
            logger.info(
                f"Saved 3D keypoint visualization: {os.path.join(save_dir, f'{image_name}_3d_kpt_visualization_{pid}.png')}"
            )

        # 综合可视化
        if cfg.get("save_together", False):
            together_img = visualize_sample_together(
                img_cv2=img_cv2,
                outputs=outputs,
                faces=faces,
                visualizer=visualizer,
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{image_name}_together_visualization.png"),
                together_img,
            )
            logger.info(
                f"Saved together visualization: {os.path.join(save_dir, f'{image_name}_together_visualization.png')}"
            )
