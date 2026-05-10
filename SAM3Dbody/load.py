#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/load.py
Project: /workspace/code/SAM3Dbody
Created Date: Friday January 23rd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 23rd 2026 4:50:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}



def load_frames(video_dir: Path) -> List[np.ndarray]:
	"""Load all frames from a video file and return RGB numpy arrays.

	Args:
		video_dir: Video file path, or a directory containing video files.

	Returns:
		A list of frames in RGB format.
	"""
	video_path = Path(video_dir)

	# If a directory is provided, pick the first supported video file.
	if video_path.is_dir():
		candidates = sorted(
			[
				p
				for p in video_path.iterdir()
				if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
			]
		)
		if not candidates:
			logger.warning("No video file found in directory: %s", video_path)
			return []
		video_path = candidates[0]

	if not video_path.is_file():
		logger.warning("Video path does not exist: %s", video_path)
		return []

	if video_path.suffix.lower() not in VIDEO_SUFFIXES:
		logger.warning("Unsupported video suffix: %s", video_path.suffix)
		return []

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		logger.error("Failed to open video: %s", video_path)
		return []

	frames: List[np.ndarray] = []
	try:
		while True:
			ret, frame_bgr = cap.read()
			if not ret:
				break
			frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
	finally:
		cap.release()

	logger.info("Loaded %d frames from: %s", len(frames), video_path)
	return frames
