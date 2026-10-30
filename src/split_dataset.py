#!/usr/bin/env python3
"""
Milestone 1: Partition competition tasks.jsonl by repository to prevent data leakage
and create a strict Out-of-Distribution (OOD) validation split for the Paper Track.

Default split:
  - Training & Synthesis Pool: fastapi/fastapi, Textualize/rich, psf/requests
  - Out-of-Distribution Validation: encode/httpx (Holdout)
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("split_dataset")


def split_tasks(
    tasks_path: Path,
    output_dir: Path,
    ood_repos: Tuple[str, ...] = ("requests", "httpx"),
) -> Tuple[List[Dict], List[Dict]]:
    """
    Partitions tasks into train_tasks.jsonl and val_ood_tasks.jsonl.

    Args:
        tasks_path: Path to raw tasks.jsonl.
        output_dir: Directory where splits will be written.
        ood_repos: Tuple of repository substrings to designate as OOD holdout.
                   Defaults to ("requests", "httpx") to provide 14 holdout tasks.
    """
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"Tasks file not found at: {tasks_path}. "
            "Please ensure competition data is downloaded into data/raw/."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train_tasks.jsonl"
    val_ood_file = output_dir / "val_ood_tasks.jsonl"

    train_tasks: List[Dict] = []
    val_ood_tasks: List[Dict] = []

    train_repo_counter: Counter = Counter()
    val_repo_counter: Counter = Counter()
    unique_commits: Set[str] = set()

    with tasks_path.open("r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                task = json.loads(line_str)
            except json.JSONDecodeError as err:
                logger.warning("Skipping invalid JSON on line %d: %s", line_num, err)
                continue

            repo = task.get("repo", "")
            base_commit = task.get("base_commit", "")
            if base_commit:
                unique_commits.add(base_commit)

            # Check if repo belongs to holdout
            is_ood = any(ood.lower() in repo.lower() for ood in ood_repos)

            if is_ood:
                val_ood_tasks.append(task)
                val_repo_counter[repo] += 1
            else:
                train_tasks.append(task)
                train_repo_counter[repo] += 1

    # Write train split
    with train_file.open("w", encoding="utf-8") as f_train:
        for t in train_tasks:
            f_train.write(json.dumps(t) + "\n")

    # Write validation OOD split
    with val_ood_file.open("w", encoding="utf-8") as f_val:
        for t in val_ood_tasks:
            f_val.write(json.dumps(t) + "\n")

    total_tasks = len(train_tasks) + len(val_ood_tasks)
    logger.info("=== Dataset Partition Complete ===")
    logger.info("Total tasks parsed: %d (across %d unique base commits)", total_tasks, len(unique_commits))
    logger.info("Training Pool (D_train): %d tasks -> %s", len(train_tasks), train_file)
    for repo, count in sorted(train_repo_counter.items()):
        logger.info("  * %s: %d tasks", repo, count)

    logger.info("OOD Validation Pool (D_val-ood): %d tasks -> %s", len(val_ood_tasks), val_ood_file)
    for repo, count in sorted(val_repo_counter.items()):
        logger.info("  * %s: %d tasks", repo, count)

    return train_tasks, val_ood_tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 1: Partition competition tasks.jsonl into Train and OOD Holdout splits."
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("data/raw/tasks.jsonl"),
        help="Path to the official tasks.jsonl file (default: data/raw/tasks.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to save train and val_ood splits (default: data/splits)",
    )
    parser.add_argument(
        "--ood-repos",
        nargs="+",
        default=["requests", "httpx"],
        help="Repository substrings to isolate as OOD holdout (default: requests httpx)",
    )
    args = parser.parse_args()

    split_tasks(
        tasks_path=args.tasks_path,
        output_dir=args.output_dir,
        ood_repos=tuple(args.ood_repos),
    )


if __name__ == "__main__":
    main()
