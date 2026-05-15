#!/usr/bin/env python3
import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


FRAME_FILE_RE = re.compile(r"^(\d+)_sam3d_body\.npz$")


@dataclass
class VideoCheckResult:
	split: str
	video_name: str
	video_path: Path
	infer_dir: Path
	frame_count: int
	infer_frame_count: int
	missing_count: int
	extra_count: int
	missing_examples: List[int]
	extra_examples: List[int]
	infer_dir_exists: bool
	none_file_exists: bool
	none_count: int
	none_examples: List[int]
	none_out_of_range_count: int
	none_out_of_range_examples: List[int]
	none_not_missing_count: int
	none_not_missing_examples: List[int]
	missing_not_in_none_count: int
	missing_not_in_none_examples: List[int]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Check SAM3D outputs against source video frames, and validate "
			"none_detected_frames.txt if it exists."
		)
	)
	parser.add_argument(
		"--video-root",
		type=Path,
		default=Path("/work/SKIING/chenkaixu/MAC_ACM_MM/data/video"),
		help="Path to video root directory, e.g. data/video",
	)
	parser.add_argument(
		"--infer-root",
		type=Path,
		default=Path("/work/SKIING/chenkaixu/MAC_ACM_MM/data/sam3d_body/inference"),
		help="Path to SAM3D inference root directory, e.g. data/sam3d_body/inference",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=["train", "val"],
		help="Data splits to check. Default: train val test",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=10,
		help="How many sample indices to print for each mismatch type",
	)
	parser.add_argument(
		"--max-videos",
		type=int,
		default=0,
		help="Limit number of videos per split for quick debug (0 means no limit)",
	)
	parser.add_argument(
		"--show-all",
		action="store_true",
		help="Print all videos, including those without mismatch",
	)
	parser.add_argument(
		"--csv",
		type=Path,
		default=Path("/work/SKIING/chenkaixu/MAC_ACM_MM/MAC_PyTorch/analysis/sam3d_body_check_results.csv"),
		help="Optional path to save detailed results as CSV",
	)
	return parser.parse_args()


def get_video_frame_count(video_path: Path) -> int:
	try:
		import cv2
	except ImportError as exc:
		raise RuntimeError(
			"OpenCV is required. Install with: pip install opencv-python"
		) from exc

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		cap.release()
		raise RuntimeError(f"Failed to open video: {video_path}")

	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()
	return frame_count


def collect_infer_frames(infer_dir: Path) -> Set[int]:
	frame_ids: Set[int] = set()
	for npz_path in infer_dir.glob("*.npz"):
		match = FRAME_FILE_RE.match(npz_path.name)
		if match is None:
			continue
		frame_ids.add(int(match.group(1)))
	return frame_ids


def parse_none_frames(none_file: Path) -> Set[int]:
	frame_ids: Set[int] = set()
	if not none_file.exists():
		return frame_ids

	with none_file.open("r", encoding="utf-8") as f:
		for line in f:
			text = line.strip()
			if not text:
				continue
			parts = re.split(r"[\s,]+", text)
			for p in parts:
				if not p:
					continue
				try:
					frame_ids.add(int(p))
				except ValueError:
					continue
	return frame_ids


def check_one_video(
	split: str,
	video_path: Path,
	infer_root: Path,
	sample_size: int,
) -> VideoCheckResult:
	frame_count = get_video_frame_count(video_path)
	infer_dir = infer_root / split / video_path.name

	if not infer_dir.exists():
		missing_examples = list(range(min(sample_size, max(0, frame_count))))
		return VideoCheckResult(
			split=split,
			video_name=video_path.name,
			video_path=video_path,
			infer_dir=infer_dir,
			frame_count=frame_count,
			infer_frame_count=0,
			missing_count=frame_count,
			extra_count=0,
			missing_examples=missing_examples,
			extra_examples=[],
			infer_dir_exists=False,
			none_file_exists=False,
			none_count=0,
			none_examples=[],
			none_out_of_range_count=0,
			none_out_of_range_examples=[],
			none_not_missing_count=0,
			none_not_missing_examples=[],
			missing_not_in_none_count=frame_count,
			missing_not_in_none_examples=missing_examples,
		)

	infer_frames = collect_infer_frames(infer_dir)
	expected = set(range(frame_count))

	missing = sorted(expected - infer_frames)
	extra = sorted(infer_frames - expected)

	none_file = infer_dir / "none_detected_frames.txt"
	none_frames = parse_none_frames(none_file)
	none_exists = none_file.exists()

	none_out_of_range = sorted(idx for idx in none_frames if idx < 0 or idx >= frame_count)
	none_not_missing = sorted(idx for idx in none_frames if idx in infer_frames)
	missing_not_in_none = sorted(idx for idx in expected - infer_frames - none_frames)

	return VideoCheckResult(
		split=split,
		video_name=video_path.name,
		video_path=video_path,
		infer_dir=infer_dir,
		frame_count=frame_count,
		infer_frame_count=len(infer_frames),
		missing_count=len(missing),
		extra_count=len(extra),
		missing_examples=missing[:sample_size],
		extra_examples=extra[:sample_size],
		infer_dir_exists=True,
		none_file_exists=none_exists,
		none_count=len(none_frames),
		none_examples=sorted(none_frames)[:sample_size],
		none_out_of_range_count=len(none_out_of_range),
		none_out_of_range_examples=none_out_of_range[:sample_size],
		none_not_missing_count=len(none_not_missing),
		none_not_missing_examples=none_not_missing[:sample_size],
		missing_not_in_none_count=len(missing_not_in_none),
		missing_not_in_none_examples=missing_not_in_none[:sample_size],
	)


def gather_videos(video_root: Path, split: str) -> List[Path]:
	split_dir = video_root / split
	if not split_dir.exists():
		return []
	return sorted(split_dir.glob("*.mp4"))


def has_any_issue(result: VideoCheckResult) -> bool:
	if not result.infer_dir_exists:
		return True
	return any(
		[
			result.missing_count > 0,
			result.extra_count > 0,
			result.none_out_of_range_count > 0,
			result.none_not_missing_count > 0,
			result.missing_not_in_none_count > 0,
		]
	)


def write_csv(results: List[VideoCheckResult], csv_path: Path) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"split",
				"video_name",
				"frame_count",
				"infer_frame_count",
				"missing_count",
				"extra_count",
				"infer_dir_exists",
				"none_file_exists",
				"none_count",
				"none_out_of_range_count",
				"none_not_missing_count",
				"missing_not_in_none_count",
				"missing_examples",
				"extra_examples",
				"none_examples",
				"none_out_of_range_examples",
				"none_not_missing_examples",
				"missing_not_in_none_examples",
			]
		)
		for r in results:
			writer.writerow(
				[
					r.split,
					r.video_name,
					r.frame_count,
					r.infer_frame_count,
					r.missing_count,
					r.extra_count,
					int(r.infer_dir_exists),
					int(r.none_file_exists),
					r.none_count,
					r.none_out_of_range_count,
					r.none_not_missing_count,
					r.missing_not_in_none_count,
					" ".join(map(str, r.missing_examples)),
					" ".join(map(str, r.extra_examples)),
					" ".join(map(str, r.none_examples)),
					" ".join(map(str, r.none_out_of_range_examples)),
					" ".join(map(str, r.none_not_missing_examples)),
					" ".join(map(str, r.missing_not_in_none_examples)),
				]
			)


def print_result_line(r: VideoCheckResult) -> None:
	status = "OK"
	if has_any_issue(r):
		status = "MISMATCH"

	print(
		f"[{status}] {r.split}/{r.video_name} | "
		f"video_frames={r.frame_count}, infer_npz={r.infer_frame_count}, "
		f"missing={r.missing_count}, extra={r.extra_count}, "
		f"none_file={int(r.none_file_exists)}, none_count={r.none_count}"
	)

	if not r.infer_dir_exists:
		print(f"  infer dir missing: {r.infer_dir}")
		return

	if r.missing_examples:
		print(f"  missing sample: {r.missing_examples}")
	if r.extra_examples:
		print(f"  extra sample:   {r.extra_examples}")

	if r.none_file_exists and r.none_examples:
		print(f"  none sample:    {r.none_examples}")

	if r.none_out_of_range_examples:
		print(f"  none out-of-range sample: {r.none_out_of_range_examples}")
	if r.none_not_missing_examples:
		print(f"  none-not-missing sample:  {r.none_not_missing_examples}")
	if r.missing_not_in_none_examples:
		print(f"  missing-not-in-none sample: {r.missing_not_in_none_examples}")


def summarize(results: List[VideoCheckResult]) -> Dict[str, int]:
	summary: Dict[str, int] = {
		"videos_total": len(results),
		"videos_ok": 0,
		"videos_mismatch": 0,
		"videos_infer_dir_missing": 0,
		"videos_none_file_exists": 0,
		"missing_frames_total": 0,
		"extra_frames_total": 0,
		"none_frames_total": 0,
		"none_out_of_range_total": 0,
		"none_not_missing_total": 0,
		"missing_not_in_none_total": 0,
	}
	for r in results:
		mismatch = has_any_issue(r)
		if mismatch:
			summary["videos_mismatch"] += 1
		else:
			summary["videos_ok"] += 1

		if not r.infer_dir_exists:
			summary["videos_infer_dir_missing"] += 1
		if r.none_file_exists:
			summary["videos_none_file_exists"] += 1

		summary["missing_frames_total"] += r.missing_count
		summary["extra_frames_total"] += r.extra_count
		summary["none_frames_total"] += r.none_count
		summary["none_out_of_range_total"] += r.none_out_of_range_count
		summary["none_not_missing_total"] += r.none_not_missing_count
		summary["missing_not_in_none_total"] += r.missing_not_in_none_count
	return summary


def main() -> None:
	args = parse_args()
	video_root: Path = args.video_root
	infer_root: Path = args.infer_root
	splits: List[str] = args.splits

	if not video_root.exists():
		raise FileNotFoundError(f"video_root not found: {video_root}")
	if not infer_root.exists():
		raise FileNotFoundError(f"infer_root not found: {infer_root}")

	all_results: List[VideoCheckResult] = []

	for split in splits:
		videos = gather_videos(video_root, split)
		if not videos:
			print(f"[WARN] No videos found in split: {split}")
			continue

		if args.max_videos > 0:
			videos = videos[: args.max_videos]

		print(f"\n=== Checking split: {split} (videos={len(videos)}) ===")

		for i, video_path in enumerate(videos, start=1):
			if i % 100 == 0:
				print(f"Progress {split}: {i}/{len(videos)}")

			result = check_one_video(
				split=split,
				video_path=video_path,
				infer_root=infer_root,
				sample_size=args.sample_size,
			)
			all_results.append(result)

			if args.show_all or has_any_issue(result):
				print_result_line(result)

	summary = summarize(all_results)
	print("\n=== Summary ===")
	for k, v in summary.items():
		print(f"{k}: {v}")

	if args.csv is not None:
		write_csv(all_results, args.csv)
		print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
	main()
