#!/usr/bin/env python3
"""
Milestone 2: Verification harness comparing Free-Form LLM Judges (V_free),
GraphForge Evidence-Anchored Rubrics (V_graph), and Deterministic Test Execution (V_exec).
"""

import argparse
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rubric_verifier")


@dataclass
class VerificationMetrics:
    precision: float
    recall: float
    f1: float
    false_positive_drift_rate: float
    cohens_kappa: float


def build_v_free_prompt(issue_description: str, candidate_patch: str) -> str:
    """Prompt for Baseline V_free (Unanchored LLM Judge)."""
    return f"""You are a code review judge. Given the issue description and proposed git diff,
decide whether the patch resolves the issue without introducing syntax or logic regressions.

### Issue Description
{issue_description}

### Candidate Git Patch
```diff
{candidate_patch}
```

Respond with a JSON object: {{"reasoning": "...", "verdict": "PASS" | "FAIL"}}
"""


def build_v_graph_prompt(
    issue_description: str,
    candidate_patch: str,
    subgraph_evidence: Dict[str, Any],
    node_rubrics: List[str],
) -> str:
    """Prompt for GraphForge V_graph (Evidence-Graph Anchored Rubric Judge)."""
    nodes_summary = "\n".join(
        f"- Node `{n.get('id')}` (file: {n.get('file', n.get('file_path', 'N/A'))}, kind: {n.get('type', n.get('kind', 'N/A'))})"
        for n in subgraph_evidence.get("nodes", [])
    )
    edges_summary = "\n".join(
        f"- `{e.get('source')}` -> `{e.get('target')}` ({e.get('type', 'dependency')})"
        for e in subgraph_evidence.get("edges", [])
    )
    rubric_items = "\n".join(f"{i+1}. {r}" for i, r in enumerate(node_rubrics))

    return f"""You are an evidence-anchored software verification judge (RepoGraphForge V_graph).
Evaluate the candidate patch strictly against the repository dependency subgraph and node-anchored rubrics.

### Issue Description
{issue_description}

### Ground-Truth Repository Dependency Subgraph
Nodes:
{nodes_summary}

Directed Edges (Call / Import / Inheritance Contracts):
{edges_summary}

### Evidence-Anchored Verification Rubric
{rubric_items}

### Candidate Git Patch
```diff
{candidate_patch}
```

Check every rubric item against the subgraph contracts. If ANY caller/callee contract is broken or incomplete, output FAIL.
Respond with JSON: {{"item_scores": [...], "verdict": "PASS" | "FAIL"}}
"""


def run_v_exec_oracle(
    repo_checkout_dir: Path,
    candidate_patch: str,
    test_patch: str,
    pytest_target: str = "tests",
    timeout_sec: int = 60,
) -> bool:
    """
    Deterministic Oracle (V_exec): Applies candidate_patch and test_patch
    inside a clean checkout of the repository at base_commit and executes pytest.
    """
    with (
        tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as cand_f,
        tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as test_f,
    ):
        cand_f.write(candidate_patch)
        test_f.write(test_patch)
        cand_path, test_path = Path(cand_f.name), Path(test_f.name)

    try:
        # Reset checkout cleanly
        subprocess.run(["git", "reset", "--hard"], cwd=repo_checkout_dir, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_checkout_dir, check=True, capture_output=True)

        # Apply candidate patch & official verification test_patch
        res_cand = subprocess.run(["git", "apply", str(cand_path)], cwd=repo_checkout_dir, capture_output=True)
        if res_cand.returncode != 0:
            return False

        res_test = subprocess.run(["git", "apply", str(test_path)], cwd=repo_checkout_dir, capture_output=True)
        if res_test.returncode != 0:
            return False

        # Execute pytest suite
        pytest_run = subprocess.run(
            ["pytest", "-q", pytest_target],
            cwd=repo_checkout_dir,
            capture_output=True,
            timeout=timeout_sec,
        )
        return pytest_run.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        cand_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)


def compute_alignment_metrics(
    predictions: List[bool], oracle_labels: List[bool]
) -> VerificationMetrics:
    """
    Compute Table 1 Paper Metrics comparing a proxy verifier against V_exec.
    Includes Precision, Recall, F1, False-Positive Drift Rate, and Cohen's Kappa.
    """
    assert len(predictions) == len(oracle_labels) and len(predictions) > 0
    tp = sum(1 for p, y in zip(predictions, oracle_labels) if p and y)
    fp = sum(1 for p, y in zip(predictions, oracle_labels) if p and not y)
    tn = sum(1 for p, y in zip(predictions, oracle_labels) if not p and not y)
    fn = sum(1 for p, y in zip(predictions, oracle_labels) if not p and y)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # False-Positive Sycophantic Patch Drift Rate: FP / (FP + TN)
    fp_drift_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Cohen's Kappa
    n = len(predictions)
    p_o = (tp + tn) / n
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    p_e = p_yes + p_no
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0

    return VerificationMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_drift_rate=round(fp_drift_rate, 4),
        cohens_kappa=round(kappa, 4),
    )


if __name__ == "__main__":
    # Smoke-test with sample predictions to verify metrics
    sample_oracle = [True, False, False, True, False, False, True, False]
    sample_v_free = [True, True, False, True, True, False, False, True]    # High FP drift
    sample_v_graph = [True, False, False, True, False, False, True, True]  # Anchored to AST

    print("=== Table 1: Verifier Alignment vs. Deterministic Execution Oracle (V_exec) ===")
    print("V_free  :", compute_alignment_metrics(sample_v_free, sample_oracle))
    print("V_graph :", compute_alignment_metrics(sample_v_graph, sample_oracle))
