#!/usr/bin/env python3
"""Split files by counts or by number of parts, and optionally save mapping files.

Example:
    # Split into 10 parts and export mapping files
    python pegasus/split_files_by_type_counts.py \
        --source-root /mnt/code-luoxi-pegasus/MAC_ACM_MM/data/video/train \
        --num-splits 10 \
        --output-root /mnt/code-luoxi-pegasus/MAC_ACM_MM/data/video/train_split_map \
        --shuffle --seed 42

    # Split by fixed counts and copy files
    python pegasus/split_files_by_type_counts.py \
        --source-root /mnt/code-luoxi-pegasus/MAC_ACM_MM/data/video \
        --split train=30000,val=5000,test=5000 \
        --output-root /mnt/code-luoxi-pegasus/MAC_ACM_MM/data/video_split \
        --mode copy --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_split_arg(split_arg: str) -> List[Tuple[str, int]]:
    """Parse split config like 'train=100,val=20,test=30'."""
    items: List[Tuple[str, int]] = []
    for raw in split_arg.split(","):
        token = raw.strip()
        if not token:
            continue

        if "=" in token:
            name, cnt = token.split("=", 1)
        elif ":" in token:
            name, cnt = token.split(":", 1)
        else:
            raise ValueError(f"Invalid split token: {token}")

        split_name = name.strip()
        if not split_name:
            raise ValueError(f"Empty split name in token: {token}")

        count = int(cnt.strip())
        if count < 0:
            raise ValueError(f"Count must be >= 0 in token: {token}")
        items.append((split_name, count))

    if not items:
        raise ValueError("No valid split items parsed")
    return items


def collect_files(source_root: Path, exts: List[str]) -> List[Path]:
    """Collect all files under source_root matching extensions."""
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    files: List[Path] = []
    for p in source_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_set:
            files.append(p)
    return sorted(files)


def to_unique_name(path: Path, base: Path) -> str:
    """Convert relative path to a safe unique file name."""
    rel = path.relative_to(base)
    return "__".join(rel.parts)


def build_assignments(
    files: List[Path],
    split_items: List[Tuple[str, int]],
    allow_remainder: bool,
    remainder_name: str,
) -> Dict[str, List[Path]]:
    """Build file assignments by ordered slices."""
    total = len(files)
    need = sum(c for _, c in split_items)

    if need > total:
        raise ValueError(f"Requested {need} files but only found {total}")

    if need < total and not allow_remainder:
        raise ValueError(
            f"Requested {need} files but found {total}. "
            "Use --allow-remainder to keep extras in a remainder split."
        )

    assignments: Dict[str, List[Path]] = {}
    start = 0
    for split_name, cnt in split_items:
        end = start + cnt
        assignments[split_name] = files[start:end]
        start = end

    if start < total and allow_remainder:
        assignments[remainder_name] = files[start:]

    return assignments


def build_equal_assignments(
    files: List[Path],
    num_splits: int,
    split_prefix: str,
) -> Dict[str, List[Path]]:
    """Build near-even assignments across num_splits parts."""
    if num_splits <= 0:
        raise ValueError("num_splits must be > 0")

    total = len(files)
    base = total // num_splits
    rem = total % num_splits

    assignments: Dict[str, List[Path]] = {}
    start = 0
    for idx in range(num_splits):
        cnt = base + (1 if idx < rem else 0)
        end = start + cnt
        split_name = f"{split_prefix}_{idx:02d}"
        assignments[split_name] = files[start:end]
        start = end
    return assignments


def save_mapping_files(
    assignments: Dict[str, List[Path]],
    source_root: Path,
    output_root: Path,
    dry_run: bool,
) -> None:
    """Save mapping json and one txt list for each split."""
    mapping_obj = {
        split_name: [str(p.relative_to(source_root)) for p in paths]
        for split_name, paths in assignments.items()
    }
    mapping_json = output_root / "mapping.json"

    print(f"mapping_json: {mapping_json}")
    if dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    with mapping_json.open("w", encoding="utf-8") as f:
        json.dump(mapping_obj, f, ensure_ascii=False, indent=2)

    for split_name, rel_paths in mapping_obj.items():
        txt_path = output_root / f"{split_name}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for rel in rel_paths:
                f.write(f"{rel}\n")


def execute_split(
    assignments: Dict[str, List[Path]],
    source_root: Path,
    output_root: Path,
    mode: str,
    dry_run: bool,
) -> None:
    """Execute file operations for all splits."""
    if mode == "none":
        return

    for split_name, file_list in assignments.items():
        out_dir = output_root / split_name
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{split_name}] {len(file_list)} files")
        for src in file_list:
            dst_name = to_unique_name(src, source_root)
            dst = out_dir / dst_name

            if dry_run:
                continue

            if mode == "copy":
                shutil.copy2(src, dst)
            elif mode == "move":
                shutil.move(str(src), str(dst))
            elif mode == "symlink":
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split files into type folders by specified counts."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Input root folder")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root folder. Default: <source-root>_split",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split counts, e.g. train=100,val=20,test=30",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=None,
        help="Split into N near-even parts, e.g. 10",
    )
    parser.add_argument(
        "--split-prefix",
        type=str,
        default="part",
        help="Name prefix used with --num-splits, default: part",
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=[".mp4"],
        help="Target file extensions. Default: .mp4",
    )
    parser.add_argument(
        "--mode",
        choices=["none", "copy", "move", "symlink"],
        default="none",
        help="none only writes mapping files; others materialize files",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument(
        "--allow-remainder",
        action="store_true",
        help="If split total is smaller than file count, keep remaining files",
    )
    parser.add_argument(
        "--remainder-name",
        type=str,
        default="remaining",
        help="Split name for remaining files",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output root before writing",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    output_root = args.output_root.resolve() if args.output_root else Path(f"{source_root}_split")

    if (args.split is None) == (args.num_splits is None):
        raise ValueError("Use exactly one of --split or --num-splits")

    files = collect_files(source_root, args.ext)

    if not files:
        raise RuntimeError("No matching files found")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)

    if args.split is not None:
        split_items = parse_split_arg(args.split)
        assignments = build_assignments(
            files=files,
            split_items=split_items,
            allow_remainder=args.allow_remainder,
            remainder_name=args.remainder_name,
        )
    else:
        assignments = build_equal_assignments(
            files=files,
            num_splits=args.num_splits,
            split_prefix=args.split_prefix,
        )

    print(f"source_root: {source_root}")
    print(f"output_root: {output_root}")
    print(f"total_files: {len(files)}")
    print(f"mode: {args.mode}, shuffle: {args.shuffle}, dry_run: {args.dry_run}")

    for split_name, file_list in assignments.items():
        print(f"[{split_name}] {len(file_list)} files")

    if args.clear_output and not args.dry_run and output_root.exists():
        shutil.rmtree(output_root)

    save_mapping_files(
        assignments=assignments,
        source_root=source_root,
        output_root=output_root,
        dry_run=args.dry_run,
    )

    execute_split(
        assignments=assignments,
        source_root=source_root,
        output_root=output_root,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    print("Done.")


if __name__ == "__main__":
    main()
