#!/usr/bin/env python3
"""
Milestone 2: GraphForge Evidence Rubric (R_G) Generator.
Supports two backends:
  1. 'vllm' (default on Kaggle): Queries local Gemma 4 (31B/12B) via http://127.0.0.1:8000/v1.
     - 100% offline, zero rate limits, zero API keys.
  2. 'gemini': Queries Google GenAI API when external network & API key are available.
"""

import argparse
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rubric_generator")

RUBRIC_SYSTEM_PROMPT = """You are an expert software verification architect implementing RepoGraphForge.
Analyze the provided bug problem statement and reference gold fix diff.
Generate a concise, verifiable 3 to 5 point Evidence Rubric (R_G) anchored to structural code contracts.

Format your output strictly as a JSON object:
{
  "target_symbols": ["module.class.func", ...],
  "structural_contracts": [
    "Contract 1: Specifying parameter/type invariance...",
    "Contract 2: Specifying caller/callee contract...",
    "Contract 3: Specifying error/exception handling or regression invariant..."
  ]
}
"""


def generate_single_rubric_vllm(
    task: Dict[str, Any],
    model_name: str = "gemma-4-31b-it-qat-w4a16-ct",
    base_url: str = "http://127.0.0.1:8000/v1",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generates rubric via local vLLM server running Gemma 4 (zero rate limits)."""
    instance_id = task.get("instance_id", "")
    repo = task.get("repo", "")
    problem = task.get("problem_statement", "")[:2000]
    patch = task.get("patch", "")[:2500]

    user_prompt = (
        f"### Task ID: {instance_id}\n"
        f"### Repo: {repo}\n\n"
        f"### Problem Statement:\n{problem}\n\n"
        f"### Gold Patch Diff:\n{patch}\n"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                contracts = parsed.get("structural_contracts", [])
                symbols = parsed.get("target_symbols", [])
                if contracts and symbols:
                    return {
                        "instance_id": instance_id,
                        "repo": repo,
                        "target_symbols": symbols,
                        "evidence_rubric": contracts,
                    }
        except Exception as e:
            time.sleep(2)
            logger.warning("[%s] vLLM retry %d/%d: %s", instance_id, attempt + 1, max_retries, e)

    return {"instance_id": instance_id, "error": "vllm_failed"}


def generate_single_rubric_gemini(
    client: Any,
    task: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 4,
) -> Dict[str, Any]:
    """Generates rubric via Google GenAI SDK with exponential backoff."""
    from google.genai import types

    instance_id = task.get("instance_id", "")
    repo = task.get("repo", "")
    problem = task.get("problem_statement", "")[:2500]
    patch = task.get("patch", "")[:3500]

    user_prompt = (
        f"### Task ID: {instance_id}\n"
        f"### Repo: {repo}\n\n"
        f"### Problem Statement:\n{problem}\n\n"
        f"### Gold Patch Diff:\n{patch}\n"
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RUBRIC_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            parsed = json.loads(response.text)
            contracts = parsed.get("structural_contracts", [])
            symbols = parsed.get("target_symbols", [])

            if contracts and symbols:
                return {
                    "instance_id": instance_id,
                    "repo": repo,
                    "target_symbols": symbols,
                    "evidence_rubric": contracts,
                }
            raise ValueError("Model returned empty contracts or symbols.")

        except Exception as e:
            wait_time = (attempt + 1) * 15
            logger.warning("[%s] Rate notice: %s. Pausing %ds...", instance_id, e, wait_time)
            time.sleep(wait_time)

    return {"instance_id": instance_id, "error": "gemini_rate_limited"}


def run_rubric_generation(
    tasks_file: Path,
    output_file: Path,
    backend: str = "vllm",
    model_name: str = "gemma-4-31b-it-qat-w4a16-ct",
    vllm_url: str = "http://127.0.0.1:8000/v1",
    delay_sec: float = 0.0,
    limit: Optional[int] = None,
) -> None:
    """Batch generates rubrics for all tasks with smart resume."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    gemini_client = None
    if backend == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        gemini_client = genai.Client(api_key=api_key)

    already_done: Set[str] = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if item.get("evidence_rubric") and not item.get("error"):
                            already_done.add(item["instance_id"])
                    except Exception:
                        pass

    logger.info("Existing verified rubrics: %d", len(already_done))

    tasks_to_process = []
    with tasks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if t.get("instance_id") not in already_done:
                    tasks_to_process.append(t)

    if limit:
        tasks_to_process = tasks_to_process[:limit]

    logger.info("Tasks to process: %d (backend: %s)", len(tasks_to_process), backend)

    with output_file.open("a", encoding="utf-8") as out_f:
        for i, task in enumerate(tasks_to_process, start=1):
            if backend == "vllm":
                res = generate_single_rubric_vllm(task, model_name=model_name, base_url=vllm_url)
            else:
                res = generate_single_rubric_gemini(gemini_client, task, model_name=model_name)

            if not res.get("error"):
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
                logger.info("[%d/%d] Generated rubric for %s", i, len(tasks_to_process), task.get("instance_id"))
            else:
                logger.error("[%d/%d] Skipped %s: %s", i, len(tasks_to_process), task.get("instance_id"), res.get("error"))

            if delay_sec > 0:
                time.sleep(delay_sec)

    logger.info("Completed rubric generation -> %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2: GraphForge Evidence Rubric Generator.")
    parser.add_argument("--tasks-file", type=Path, default=Path("data/raw/tasks.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/splits/evidence_rubrics.jsonl"))
    parser.add_argument("--backend", choices=["vllm", "gemini"], default="vllm")
    parser.add_argument("--model-name", type=str, default="gemma-4-31b-it-qat-w4a16-ct")
    parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_rubric_generation(
        tasks_file=args.tasks_file,
        output_file=args.output_file,
        backend=args.backend,
        model_name=args.model_name,
        vllm_url=args.vllm_url,
        delay_sec=args.delay_sec,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
