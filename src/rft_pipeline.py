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
    best_of_n: bool = True,
) -> Dict[str, Any]:
    """
    Filters candidate rollouts into three verifier regimes:
      - V_free (unanchored judge)
      - V_graph (GraphForge evidence rubric judge)
      - Hybrid (V_graph + V_exec heuristic)

    If best_of_n=True, selects the shortest passing trajectory per task
    to minimize turn count, context bloat, and agent timeouts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    v_free_file = output_dir / "rft_v_free_accepted.jsonl"
    v_graph_file = output_dir / "rft_v_graph_accepted.jsonl"
    v_hybrid_file = output_dir / "rft_hybrid_accepted.jsonl"

    from collections import defaultdict
    candidates_by_task = defaultdict(list)

    total_candidates = 0
    with rollouts_file.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            entry = json.loads(line)
            syn_id = entry.get("synthetic_id", f"syn_{total_candidates}")
            candidates_by_task[syn_id].append(entry)
            total_candidates += 1

    selected_free = []
    selected_graph = []
    selected_hybrid = []

    for syn_id, samples in candidates_by_task.items():
        if best_of_n:
            # Pick shortest passing trajectory for each regime
            free_passes = [s for s in samples if s.get("v_free_pass")]
            graph_passes = [s for s in samples if s.get("v_graph_pass")]
            hybrid_passes = [s for s in samples if s.get("v_graph_pass") and s.get("v_exec_pass")]

            if free_passes:
                best_free = min(free_passes, key=lambda s: len(s.get("text", "")))
                selected_free.append(best_free)
            if graph_passes:
                best_graph = min(graph_passes, key=lambda s: len(s.get("text", "")))
                selected_graph.append(best_graph)
            if hybrid_passes:
                best_hybrid = min(hybrid_passes, key=lambda s: len(s.get("text", "")))
                selected_hybrid.append(best_hybrid)
        else:
            for s in samples:
                if s.get("v_free_pass"):
                    selected_free.append(s)
                if s.get("v_graph_pass"):
                    selected_graph.append(s)
                if s.get("v_graph_pass") and s.get("v_exec_pass"):
                    selected_hybrid.append(s)

    def write_dataset(entries: List[Dict[str, Any]], out_path: Path):
        with out_path.open("w", encoding="utf-8") as out_f:
            for e in entries:
                rec = {
                    "synthetic_id": e.get("synthetic_id"),
                    "sample_idx": e.get("sample_idx", 0),
                    "text": e.get("text", ""),
                }
                out_f.write(json.dumps(rec) + "\n")

    write_dataset(selected_free, v_free_file)
    write_dataset(selected_graph, v_graph_file)
    write_dataset(selected_hybrid, v_hybrid_file)

    n_tasks = len(candidates_by_task)
    avg_len_free = sum(len(e.get("text", "")) for e in selected_free) / max(1, len(selected_free))
    avg_len_graph = sum(len(e.get("text", "")) for e in selected_graph) / max(1, len(selected_graph))
    avg_len_hybrid = sum(len(e.get("text", "")) for e in selected_hybrid) / max(1, len(selected_hybrid))

    logger.info("===========================================================")
    logger.info("=== RFT BEST-OF-N VERIFIER FILTERING SUMMARY (%d Tasks) ===", n_tasks)
    logger.info("===========================================================")
    logger.info("Total Candidate Rollouts Evaluated: %d", total_candidates)
    logger.info("V_free Accepted:   %d / %d tasks (%.1f%%) | Avg Chars: %.0f", len(selected_free), n_tasks, (len(selected_free)/max(1, n_tasks))*100, avg_len_free)
    logger.info("V_graph Accepted:  %d / %d tasks (%.1f%%) | Avg Chars: %.0f", len(selected_graph), n_tasks, (len(selected_graph)/max(1, n_tasks))*100, avg_len_graph)
    logger.info("Hybrid Accepted:   %d / %d tasks (%.1f%%) | Avg Chars: %.0f", len(selected_hybrid), n_tasks, (len(selected_hybrid)/max(1, n_tasks))*100, avg_len_hybrid)
    logger.info("Saved filtered datasets to: %s", output_dir)

    return {
        "n_tasks": n_tasks,
        "total_candidates": total_candidates,
        "v_free_count": len(selected_free),
        "v_graph_count": len(selected_graph),
        "hybrid_count": len(selected_hybrid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4: RFT Candidate Rollout Filter.")
    parser.add_argument("--rollouts-file", type=Path, default=Path("data/trajectories/rft_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/trajectories/rft_splits"))
    parser.add_argument("--all-samples", action="store_true", help="Keep all accepted samples instead of best-of-N.")
    args = parser.parse_args()

    filter_rollouts(args.rollouts_file, args.output_dir, best_of_n=not args.all_samples)


if __name__ == "__main__":
    main()
