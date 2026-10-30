#!/usr/bin/env python3
"""
Milestone 4: Rejection Fine-Tuning (RFT) Pipeline.
Coordinates Best-of-N candidate rollout filtering across three verifier conditions:
  - Condition A (V_free): Unanchored judge
  - Condition B (V_graph): GraphForge evidence rubric judge
  - Condition C (Hybrid Oracle): V_exec (pytest) + V_graph

Outputs filtered datasets for comparative RFT training (Table 2 Rows 3, 4, 5).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rft_pipeline")


def filter_rollouts(
    rollouts_file: Path,
    output_dir: Path,
) -> Dict[str, int]:
    """Filters candidate rollouts into three verifier regimes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    v_free_file = output_dir / "rft_v_free_accepted.jsonl"
    v_graph_file = output_dir / "rft_v_graph_accepted.jsonl"
    v_hybrid_file = output_dir / "rft_hybrid_accepted.jsonl"

    counts = {"v_free": 0, "v_graph": 0, "hybrid": 0}

    with (
        rollouts_file.open("r", encoding="utf-8") as in_f,
        v_free_file.open("w", encoding="utf-8") as f_free,
        v_graph_file.open("w", encoding="utf-8") as f_graph,
        v_hybrid_file.open("w", encoding="utf-8") as f_hybrid,
    ):
        for line in in_f:
            if not line.strip():
                continue
            entry = json.loads(line)
            passes_free = bool(entry.get("v_free_pass", False))
            passes_graph = bool(entry.get("v_graph_pass", False))
            passes_exec = bool(entry.get("v_exec_pass", False))

            record = {
                "synthetic_id": entry.get("synthetic_id"),
                "messages": entry.get("messages", []),
                "text": entry.get("text", ""),
            }

            line_str = json.dumps(record) + "\n"

            if passes_free:
                f_free.write(line_str)
                counts["v_free"] += 1

            if passes_graph:
                f_graph.write(line_str)
                counts["v_graph"] += 1

            if passes_exec and passes_graph:
                f_hybrid.write(line_str)
                counts["hybrid"] += 1

    logger.info("=== RFT Verification Filtering Summary ===")
    logger.info("V_free Accepted:   %d -> %s", counts["v_free"], v_free_file)
    logger.info("V_graph Accepted:  %d -> %s", counts["v_graph"], v_graph_file)
    logger.info("Hybrid Accepted:   %d -> %s", counts["hybrid"], v_hybrid_file)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4: RFT Candidate Rollout Filter.")
    parser.add_argument("--rollouts-file", type=Path, default=Path("data/trajectories/rft_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/trajectories/rft_splits"))
    args = parser.parse_args()

    filter_rollouts(args.rollouts_file, args.output_dir)


if __name__ == "__main__":
    main()
