#!/usr/bin/env python3
"""
Milestone 2: Verifier Alignment & Sycophantic Patch Drift Evaluator (Table 1).
Runs V_free (unanchored judge) vs V_graph (GraphForge evidence rubric judge)
against the deterministic execution oracle (V_exec) and outputs Table 1 metrics.

Supports two backends:
  - 'vllm' (default on Kaggle): Local Gemma 4 (31B/12B) via http://127.0.0.1:8000/v1 with multi-threaded continuous batching.
  - 'gemini': Google GenAI API.
"""

import argparse
import json
import logging
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.rubric_verifier import VerificationMetrics, compute_alignment_metrics
except ImportError:
    from rubric_verifier import VerificationMetrics, compute_alignment_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_verifier_alignment")


def auto_detect_vllm_model(base_url: str = "http://127.0.0.1:8000/v1") -> str:
    """Queries vLLM /v1/models endpoint to auto-detect the served model ID string."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            model_id = data["data"][0]["id"]
            logger.info("Auto-detected vLLM served model: '%s'", model_id)
            return model_id
    except Exception as e:
        logger.warning("Could not auto-detect model from %s (%s). Using default.", url, e)
        return "gemma-4-31b-it-qat-w4a16-ct"


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Cleans markdown codeblock wrapping and repairs JSON backslashes."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text, strict=False)
    except Exception:
        pass
    try:
        fixed = re.sub(r'\\([^\'"\\/bfnrtu])', r'\\\\\1', text)
        return json.loads(fixed, strict=False)
    except Exception:
        return None


def call_vllm_chat(
    prompt: str,
    system_prompt: str,
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
) -> Optional[Dict[str, Any]]:
    """Helper to query local vLLM chat completions."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return safe_parse_json(content)
    except Exception as e:
        logger.warning("vLLM call failed: %s", e)
        return None


def judge_v_free(
    problem: str,
    patch: str,
    backend: str = "vllm",
    client: Any = None,
    model_name: str = "gemma-4-31b-it-qat-w4a16-ct",
    vllm_url: str = "http://127.0.0.1:8000/v1",
) -> bool:
    """Evaluates patch under V_free (Unanchored LLM Judge)."""
    sys_prompt = (
        "You are a strict code review judge. Given the problem statement and proposed git diff, "
        "decide whether the patch resolves the issue without introducing syntax or logic regressions. "
        "Respond with JSON: {\"reasoning\": \"...\", \"verdict\": \"PASS\" | \"FAIL\"}"
    )
    user_prompt = f"### Problem Statement:\n{problem[:2000]}\n\n### Candidate Git Patch:\n```diff\n{patch[:3000]}\n```\n"

    if backend == "vllm":
        res = call_vllm_chat(user_prompt, sys_prompt, model_name=model_name, base_url=vllm_url)
        return str(res.get("verdict", "")).upper() == "PASS" if res else False

    # Gemini fallback
    from google.genai import types
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        data = json.loads(response.text)
        return str(data.get("verdict", "")).upper() == "PASS"
    except Exception as e:
        logger.warning("V_free Gemini error: %s", e)
        return False


def judge_v_graph(
    problem: str,
    patch: str,
    rubric_items: List[str],
    backend: str = "vllm",
    client: Any = None,
    model_name: str = "gemma-4-31b-it-qat-w4a16-ct",
    vllm_url: str = "http://127.0.0.1:8000/v1",
) -> bool:
    """Evaluates patch under V_graph (GraphForge Evidence Rubric Judge)."""
    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric_items))
    sys_prompt = (
        "You are an evidence-anchored code verification judge (RepoGraphForge V_graph). "
        "Evaluate the candidate patch strictly against the provided structural evidence rubric and contracts. "
        "If ANY contract is violated, omitted, or partially implemented, output FAIL. "
        "Respond with JSON: {\"contract_evaluations\": [...], \"verdict\": \"PASS\" | \"FAIL\"}"
    )
    user_prompt = (
        f"### Problem Statement:\n{problem[:2000]}\n\n"
        f"### Evidence-Anchored Verification Rubric (R_G):\n{rubric_text}\n\n"
        f"### Candidate Git Patch:\n```diff\n{patch[:3000]}\n```\n"
    )

    if backend == "vllm":
        res = call_vllm_chat(user_prompt, sys_prompt, model_name=model_name, base_url=vllm_url)
        return str(res.get("verdict", "")).upper() == "PASS" if res else False

    # Gemini fallback
    from google.genai import types
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        data = json.loads(response.text)
        return str(data.get("verdict", "")).upper() == "PASS"
    except Exception as e:
        logger.warning("V_graph Gemini error: %s", e)
        return False


def evaluate_single_candidate(
    c: Dict[str, Any],
    backend: str,
    model_name: str,
    vllm_url: str,
    gemini_client: Any,
) -> Dict[str, Any]:
    """Evaluates a single candidate under both V_free and V_graph."""
    task_id = c["task_id"]
    c_type = c["candidate_type"]
    oracle = bool(c["oracle_label"])
    rubric = c.get("evidence_rubric", [])

    pred_free = judge_v_free(
        problem=c["problem_statement"],
        patch=c["patch"],
        backend=backend,
        client=gemini_client,
        model_name=model_name,
        vllm_url=vllm_url,
    )

    pred_graph = judge_v_graph(
        problem=c["problem_statement"],
        patch=c["patch"],
        rubric_items=rubric,
        backend=backend,
        client=gemini_client,
        model_name=model_name,
        vllm_url=vllm_url,
    )

    return {
        "task_id": task_id,
        "candidate_type": c_type,
        "oracle_label": oracle,
        "v_free": pred_free,
        "v_graph": pred_graph,
    }


def run_experiment(
    candidates_file: Path,
    output_log_file: Path,
    backend: str = "vllm",
    model_name: Optional[str] = None,
    vllm_url: str = "http://127.0.0.1:8000/v1",
    workers: int = 4,
    delay_sec: float = 0.0,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs V_free and V_graph across all candidate patches and logs Table 1 results."""
    output_log_file.parent.mkdir(parents=True, exist_ok=True)

    if backend == "vllm" and not model_name:
        model_name = auto_detect_vllm_model(vllm_url)

    gemini_client = None
    if backend == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        gemini_client = genai.Client(api_key=api_key)

    candidates: List[Dict[str, Any]] = []
    with candidates_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    if sample_limit:
        candidates = candidates[:sample_limit]

    logger.info(
        "Evaluating %d candidate patches (backend: %s, model: %s, workers: %d)...",
        len(candidates), backend, model_name, workers
    )

    start_time = time.time()
    results: List[Dict[str, Any]] = []

    if backend == "vllm" and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    evaluate_single_candidate, c, backend, model_name, vllm_url, gemini_client
                ): c
                for c in candidates
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                if i % 40 == 0 or i == len(candidates):
                    logger.info("[%d/%d] Completed in %.1fs...", i, len(candidates), time.time() - start_time)
    else:
        for i, c in enumerate(candidates, start=1):
            res = evaluate_single_candidate(c, backend, model_name, vllm_url, gemini_client)
            results.append(res)
            if i % 25 == 0 or i == len(candidates):
                logger.info("[%d/%d] Completed in %.1fs...", i, len(candidates), time.time() - start_time)
            if delay_sec > 0:
                time.sleep(delay_sec)

    v_free_preds = [r["v_free"] for r in results]
    v_graph_preds = [r["v_graph"] for r in results]
    oracle_labels = [r["oracle_label"] for r in results]

    # Compute Table 1 metrics
    m_free = compute_alignment_metrics(v_free_preds, oracle_labels)
    m_graph = compute_alignment_metrics(v_graph_preds, oracle_labels)

    table_1_summary = {
        "num_evaluated_patches": len(candidates),
        "v_free_metrics": m_free.__dict__,
        "v_graph_metrics": m_graph.__dict__,
    }

    with output_log_file.open("w", encoding="utf-8") as out_f:
        json.dump({"summary": table_1_summary, "records": results}, out_f, indent=2)

    print("\n" + "=" * 80)
    print("TABLE 1: VERIFIER ALIGNMENT & SYCOPHANTIC PATCH DRIFT ANALYSIS (PAPER TRACK)")
    print("=" * 80)
    print(f"| Verifier Signal | Precision vs V_exec | Recall vs V_exec | F1 Score | FP Drift Rate | Cohen's κ |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| **V_free (Unanchored Judge)**   | {m_free.precision:.4f} | {m_free.recall:.4f} | {m_free.f1:.4f} | {m_free.false_positive_drift_rate:.4f} | {m_free.cohens_kappa:.4f} |")
    print(f"| **V_graph (GraphForge Rubric)** | {m_graph.precision:.4f} | {m_graph.recall:.4f} | {m_graph.f1:.4f} | {m_graph.false_positive_drift_rate:.4f} | {m_graph.cohens_kappa:.4f} |")
    print("=" * 80 + "\n")

    return table_1_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2: Evaluate verifier alignment (Table 1).")
    parser.add_argument("--candidates-file", type=Path, default=Path("data/splits/candidate_eval_suite.jsonl"))
    parser.add_argument("--output-log", type=Path, default=Path("experiments/logs/verifier_alignment_table1.json"))
    parser.add_argument("--backend", choices=["vllm", "gemini"], default="vllm")
    parser.add_argument("--model-name", type=str, default=None, help="Auto-detected if vllm")
    parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    run_experiment(
        candidates_file=args.candidates_file,
        output_log_file=args.output_log,
        backend=args.backend,
        model_name=args.model_name,
        vllm_url=args.vllm_url,
        workers=args.workers,
        delay_sec=args.delay_sec,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
