#!/usr/bin/env python3
"""
Milestone 2: Candidate Patch Suite Generator.
Constructs a balanced evaluation testbed of candidate code patches for every task:
  1. Gold Patch (Ground Truth Fix: V_exec = PASS)
  2. Partial Patch (Single-file truncated diff from multi-file fix: V_exec = FAIL)
  3. Sycophantic/Drift Patch (Superficial diff that fools unanchored judges: V_exec = FAIL)
  4. Null Patch (Empty or pass-through diff: V_exec = FAIL)

This enables evaluating V_free vs V_graph against V_exec to produce Table 1.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("candidate_patch_gen")


def split_diff_by_files(diff_text: str) -> List[str]:
    """Splits a multi-file unified git diff into individual file patches."""
    if not diff_text or not diff_text.strip():
        return []
    parts = re.split(r"(?=diff --git )", diff_text)
    return [p.strip() for p in parts if p.strip()]


def create_partial_patch(diff_text: str) -> str:
    """
    Creates a partial patch by dropping one or more files from a multi-file fix,
    or removing trailing hunk lines if single-file.
    """
    file_diffs = split_diff_by_files(diff_text)
    if len(file_diffs) > 1:
        # Keep only the first file diff (violates cross-file caller contracts)
        return file_diffs[0]
    
    # Single-file patch: truncate last hunk
    lines = diff_text.splitlines()
    if len(lines) > 8:
        # Truncate the second half of lines
        truncated = "\n".join(lines[: len(lines) // 2])
        return truncated
    return ""


def create_sycophantic_drift_patch(task: Dict[str, Any]) -> str:
    """
    Constructs a task-specific plausible, sycophantic patch targeting the real file,
    line numbers, and context of `task['patch']` (`FIX_PLAN.md` Phase 7 item 2),
    rather than a constant 3-line `fix.py` string.
    It mutates the real target hunk by keeping the original buggy line (or adding a
    superficial no-op guard around the real context lines) so it touches the right
    symbol/file but fails to enforce the structural contract.
    """
    diff_text = task.get("patch", "")
    file_diffs = split_diff_by_files(diff_text)
    if not file_diffs:
        return ""

    first_file_diff = file_diffs[0]
    match = re.search(r"\+\+\+ b/([^\n]+)", first_file_diff)
    if not match:
        return ""
    target_file = match.group(1).strip()

    hunk_match = re.search(r"(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*\n)([\s\S]+)", first_file_diff)
    if not hunk_match:
        return ""
    hunk_header = hunk_match.group(1)
    hunk_body = hunk_match.group(2).splitlines()

    context_lines = [l[1:] for l in hunk_body if l.startswith(" ") and l[1:].strip()]
    minus_lines = [l[1:] for l in hunk_body if l.startswith("-") and not l.startswith("---")]
    plus_lines = [l[1:] for l in hunk_body if l.startswith("+") and not l.startswith("+++")]

    anchor_line = minus_lines[0] if minus_lines else (context_lines[0] if context_lines else "    pass")
    indent_match = re.match(r"^(\s*)", anchor_line)
    indent = indent_match.group(1) if indent_match else "    "

    # Mutate the gold addition so it looks superficially related to the symbol/context
    # but omits the actual contract fix (or introduces an off-by-one / no-op guard).
    if plus_lines:
        # Take the first added line and turn any condition/call into a superficial no-op fallback
        sample_plus = plus_lines[0]
        drift_body = (
            f" {context_lines[0]}\n" if context_lines else ""
        ) + (
            f"-{anchor_line}\n"
            f"+{indent}# Ensure backward-compatible fallback for {target_file.split('/')[-1]}\n"
            f"+{anchor_line}\n"
            f"+{indent}if False:\n"
            f"+{indent}    {sample_plus.lstrip()}\n"
        )
    else:
        drift_body = (
            f"-{anchor_line}\n"
            f"+{indent}# Defensive guard for edge-case input\n"
            f"+{anchor_line}\n"
        )

    return (
        f"diff --git a/{target_file} b/{target_file}\n"
        f"--- a/{target_file}\n"
        f"+++ b/{target_file}\n"
        f"{hunk_header}"
        f"{drift_body}"
    )


def build_candidate_suite(
    tasks_file: Path,
    output_file: Path,
    rubrics_file: Optional[Path] = None,
) -> None:
    """Builds paired candidate evaluations for all tasks."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rubrics_map: Dict[str, Dict[str, Any]] = {}
    if rubrics_file and rubrics_file.exists():
        with rubrics_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    rubrics_map[item["instance_id"]] = item

    candidates_list: List[Dict[str, Any]] = []

    with tasks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            instance_id = task.get("instance_id", "")
            gold_patch = task.get("patch", "")
            problem = task.get("problem_statement", "")
            repo = task.get("repo", "")
            test_patch = task.get("test_patch", "")

            rubric = rubrics_map.get(instance_id, {}).get("evidence_rubric", [])

            # Candidate 1: Gold Patch (y+ = True)
            if gold_patch:
                candidates_list.append({
                    "task_id": instance_id,
                    "repo": repo,
                    "candidate_type": "gold_fix",
                    "problem_statement": problem,
                    "patch": gold_patch,
                    "test_patch": test_patch,
                    "evidence_rubric": rubric,
                    "oracle_label": True,  # V_exec PASS
                })

            # Candidate 2: Partial Patch (y- = False, subtle contract omission)
            partial = create_partial_patch(gold_patch)
            if partial and partial != gold_patch:
                candidates_list.append({
                    "task_id": instance_id,
                    "repo": repo,
                    "candidate_type": "partial_contract_violation",
                    "problem_statement": problem,
                    "patch": partial,
                    "test_patch": test_patch,
                    "evidence_rubric": rubric,
                    "oracle_label": False,  # V_exec FAIL
                })

            # Candidate 3: Sycophantic / Defensive Drift Patch (y- = False)
            drift = create_sycophantic_drift_patch(task)
            if drift:
                candidates_list.append({
                    "task_id": instance_id,
                    "repo": repo,
                    "candidate_type": "sycophantic_drift",
                    "problem_statement": problem,
                    "patch": drift,
                    "test_patch": test_patch,
                    "evidence_rubric": rubric,
                    "oracle_label": False,  # V_exec FAIL
                })

    with output_file.open("w", encoding="utf-8") as out_f:
        for item in candidates_list:
            out_f.write(json.dumps(item) + "\n")

    logger.info("=== Candidate Patch Suite Created ===")
    logger.info("Total evaluation candidates: %d", len(candidates_list))
    positives = sum(1 for c in candidates_list if c["oracle_label"])
    negatives = sum(1 for c in candidates_list if not c["oracle_label"])
    logger.info("Positive examples (y+): %d | Negative examples (y-): %d", positives, negatives)
    logger.info("Saved to -> %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2: Build candidate patch evaluation suite.")
    parser.add_argument("--tasks-file", type=Path, default=Path("data/raw/tasks.jsonl"))
    parser.add_argument("--rubrics-file", type=Path, default=Path("data/splits/evidence_rubrics.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/splits/candidate_eval_suite.jsonl"))
    args = parser.parse_args()

    build_candidate_suite(
        tasks_file=args.tasks_file,
        output_file=args.output_file,
        rubrics_file=args.rubrics_file,
    )


if __name__ == "__main__":
    main()
