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

    log_dir = action_log_root / "action_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    frame_list = load_frames(video_dir)

    logger.info(
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
        logger.info(
            "Inference results already exist for %s, skipping processing.",
            rel_video,
        )
        return

    logger.info("==== Start Video: %s ====", rel_video)

    process_frame_list(
        frame_list=frame_list,
        out_dir=out_dir,
        inference_output_path=infer_dir,
        cfg=cfg,
    )

    logger.info("==== Finished Video: %s ====", rel_video)


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
    

    video_dirs = []
    
    _idx_path = cfg_dict['paths']['video_split_path'] 
    with open(_idx_path, "r") as f:
        video_name_list = [line.strip() for line in f if line.strip()]
    logger.info("[%s] Found %d video names in split file: %s", dataset_type, len(video_name_list), _idx_path)

    for name in video_name_list:
        video_path = dataset_source_root / name
        if video_path.suffix == ".mp4":
            video_dirs.append(video_path)

    if not video_dirs:
        logger.warning("[%s] No video dirs found", dataset_type)
        return
    video_dirs = sorted(video_dirs)

    logger.info("[%s] Processing %d video dirs", dataset_type, len(video_dirs))
    
    # Reverse first so newest videos are dispatched first.
    video_dirs = list(reversed(video_dirs))

    # Default behavior: run multiple workers on the first configured device.
    gpu_ids = cfg.infer.get("gpu", 0)
    
    workers_per_gpu = int(cfg.infer.get("workers_per_gpu", 1))
    if workers_per_gpu < 1:
        logger.warning("[%s] Invalid workers_per_gpu=%s, fallback to 1", dataset_type, workers_per_gpu)


    num_workers = min(workers_per_gpu, len(video_dirs))
    
    # Round-robin split keeps workload balanced while preserving newest-first scheduling.
    video_chunks: List[List[Path]] = [[] for _ in range(num_workers)]
    for idx, one_video_dir in enumerate(video_dirs):
        video_chunks[idx % num_workers].append(one_video_dir)

    cfg_dict_local = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict_local, dict):
        logger.error("[%s] Failed to convert worker config to dict", dataset_type)
        return

    logger.info(
        "[%s] Parallel mode. gpu_ids=%s, workers_per_gpu=%d, total_workers=%d, use_all_gpus=%s",
        dataset_type,
        gpu_ids,
        workers_per_gpu,
        num_workers,
    )

    task_args = []
    for worker_idx in range(num_workers):
        chunk = video_chunks[worker_idx]
        if not chunk:
            continue
        task_args.append(
            (
                dataset_type,
                worker_idx,
                chunk,
                dataset_source_root,
                dataset_vis_root,
                dataset_infer_root,
                action_log_root,
                cfg_dict_local,
            )
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(task_args)) as pool:
        pool.starmap(_process_video_chunk, task_args)

    logger.info("[%s] Finished processing", dataset_type)


def _process_video_chunk(
    dataset_type: str,
    worker_idx: int,
    video_dirs: List[Path],
    dataset_source_root: Path,
    dataset_vis_root: Path,
    dataset_infer_root: Path,
    action_log_root: Path,
    cfg_dict: dict,
) -> None:
    """Process a video chunk in one worker process."""
    cfg = OmegaConf.create(cfg_dict)

    logger.info(
        "[%s][worker-%d] Start. gpu=%s, assigned_videos=%d",
        dataset_type,
        worker_idx,
        len(video_dirs),
    )

    for one_video_dir in video_dirs:
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
                "[%s][worker-%d] Failed on action %s: %s",
                dataset_type,
                worker_idx,
                one_video_dir.name,
                exc,
            )

    logger.info("[%s][worker-%d] Finished", dataset_type, worker_idx)


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