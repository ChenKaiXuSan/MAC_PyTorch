#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Run SAM-3D-Body inference on unity image frames by action in parallel."""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from .infer import process_frame_list
from .load import load_frames

logger = logging.getLogger(__name__)


def process_single_video(
    video_dir: Path,
    source_root: Path,
    vis_root: Path,
    infer_root: Path,
    action_log_root: Path,
    cfg: DictConfig,
) -> None:
    """Process all captures in one action directory.
    
    Args:
        camera_layers: Optional list of layer indices (0-4) to filter captures.
    """
    rel_video = video_dir.relative_to(source_root)
    video_id = str(rel_video).replace("/", "__")

    log_dir = action_log_root / "action_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    action_log_file = log_dir / f"{video_id}.log"

    handler = logging.FileHandler(action_log_file, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    video_logger = logging.getLogger(f"action_{video_id}")
    video_logger.handlers.clear()
    video_logger.setLevel(logging.INFO)
    video_logger.addHandler(handler)
    video_logger.propagate = False

    video_logger.info("==== Start Video: %s ====", rel_video)

    frame_list = load_frames(video_dir)
    
    if not frame_list:
        video_logger.warning("No frames found for %s, skipping.", rel_video)
        return

    video_logger.info(
        "Processing %s, frame_count=%d",
        rel_video,
        len(frame_list),
    )

    out_dir = vis_root / rel_video
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_dir = infer_root / rel_video
    infer_dir.mkdir(parents=True, exist_ok=True)

    # 检查输出目录是否已经存在结果文件，如果存在则跳过处理
    existing_infer_files = list(infer_dir.glob("*.npz"))

    if len(existing_infer_files) == len(frame_list):
        video_logger.info(
            "Inference results already exist for %s, skipping processing.",
            rel_video,
        )
        return

    process_frame_list(
        frame_list=frame_list,
        out_dir=out_dir,
        inference_output_path=infer_dir,
        cfg=cfg,
    )

    video_logger.info("==== Finished Video: %s ====", rel_video)


def dataset_worker(
    dataset_type: str,
    source_root: Path,
    vis_root: Path,
    infer_root: Path,
    action_log_root: Path,
    cfg_dict: dict,
) -> None:
    """Worker entrypoint for processing a single dataset (train/val/test).
    
    Args:
        dataset_type: One of ["train", "val", "test"]
        camera_layers: Optional list of layer indices to filter captures.
        person_filter: Optional person filter.
        action_filter: Optional action filter.
    """
    dataset_source_root = source_root / dataset_type
    if not dataset_source_root.is_dir():
        logger.warning("[%s] Dataset dir not found: %s", dataset_type, dataset_source_root)
        return

    # Create dataset-specific output directories
    dataset_vis_root = vis_root / dataset_type
    dataset_infer_root = infer_root / dataset_type
    dataset_vis_root.mkdir(parents=True, exist_ok=True)
    dataset_infer_root.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(cfg_dict)

    logger.info("[%s] Started processing", dataset_type)

    video_dirs: list = list(dataset_source_root.glob("**/*.mp4"))  # Adjust the pattern as needed
    if not video_dirs:
        logger.warning("[%s] No video dirs found", dataset_type)
        return
    video_dirs = sorted(video_dirs)

    logger.info("[%s] Processing %d video dirs", dataset_type, len(video_dirs))
    
    # 从后往前处理，优先处理最新的视频
    for one_video_dir in video_dirs[::-1]:
        try:
            process_single_video(
                one_video_dir,
                dataset_source_root,
                dataset_vis_root,
                dataset_infer_root,
                action_log_root,
                cfg,
            )
        except Exception as exc:
            logger.error(
                "[%s] Failed on action %s: %s",
                dataset_type,
                one_video_dir.name,
                exc,
            )

    logger.info("[%s] Finished processing", dataset_type)


def normalize_gpu_ids(raw_gpu_ids) -> List[Union[int, str]]:
    """Normalize gpu config to a list of integer ids."""
    if isinstance(raw_gpu_ids, str) and raw_gpu_ids.lower() == "cpu":
        return ["cpu"]

    if isinstance(raw_gpu_ids, int):
        return [raw_gpu_ids]

    if isinstance(raw_gpu_ids, str):
        if "," in raw_gpu_ids:
            parsed_ids: List[Union[int, str]] = []
            for x in raw_gpu_ids.split(","):
                x = x.strip()
                if not x:
                    continue
                parsed_ids.append("cpu" if x.lower() == "cpu" else int(x))
            return parsed_ids
        return [int(raw_gpu_ids)]

    if isinstance(raw_gpu_ids, (list, tuple)):
        parsed_ids: List[Union[int, str]] = []
        for x in raw_gpu_ids:
            if isinstance(x, str) and x.lower() == "cpu":
                parsed_ids.append("cpu")
            else:
                parsed_ids.append(int(x))
        return parsed_ids

    return [0]

@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    source_root = Path(cfg.paths.video_path).resolve()
    result_root = Path(cfg.paths.output_path).resolve()

    vis_root = result_root / "visualization"
    infer_root = result_root / "inference"
    action_log_root = Path(cfg.paths.log_path).resolve()
    vis_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)

    logger.info("Source data root: %s", source_root)
    logger.info("Result root: %s", result_root)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        logger.error("Failed to convert config to dict")
        return

    # Run train/val/test sequentially
    dataset_types = list(cfg.infer.data_types)

    for dataset_type in dataset_types:
        logger.info("Start dataset: %s", dataset_type)
        dataset_worker(
            dataset_type,
            source_root,
            vis_root,
            infer_root,
            action_log_root,
            cfg_dict,
        )
        logger.info("Finished dataset: %s", dataset_type)

    logger.info("[SUCCESS] All datasets completed sequentially")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()

# command 
# python -m SAM3Dbody.main