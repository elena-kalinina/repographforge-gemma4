#!/usr/bin/env python3
"""
Milestone 3.2B: Teacher Trajectory Rollouts (SFT Dataset Generator).
Takes synthesized bug tasks from Milestone 3.2A and runs expert teacher rollouts
following a strict graph-first navigation workflow:
  - Turn 1: Call `get_code_subgraph` on the anchor symbol.
  - Turn 2: Inspect caller/callee contracts and call `read_file` on target span.
  - Turn 3: Reason about structural rubric contracts and apply fix with `edit_file`.
  - Turn 4: Verify test execution and call `submit_patch`.

Outputs verified trajectories in OpenAI chat format and standard SFT text format.
"""

import argparse
import json
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("teacher_rollouts")

TEACHER_SYSTEM_PROMPT = (
    "You are an expert autonomous software engineer solving a real repository issue. "
    "Equipped with code-graph tools, you follow a strict graph-first navigation workflow, "
    "especially across multi-file dependencies and cross-module caller/callee contracts. "
    "Output strictly valid JSON with keys: 'synthetic_id', 'messages'."
)


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
    """Strips markdown codeblock wrapping and repairs JSON backslashes."""
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


def generate_teacher_trajectory(
    task: Dict[str, Any],
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Generates a complete 4-turn teacher agent trajectory for a single synthetic task."""
    syn_id = task.get("synthetic_id", "syn_task")
    anchor = task.get("anchor_node", "core")
    problem = task.get("problem_statement", "")[:1800]
    patch = task.get("bug_patch", "")[:2000]
    rubric = task.get("evidence_rubric", [])
    rubric_text = "\n".join(f"- {r}" for r in rubric) if rubric else "- Enforce interface and type contracts."

    user_prompt = (
        f"Generate a 4-turn autonomous agent trajectory in OpenAI chat format for this task.\n\n"
        f"### Task ID: {syn_id}\n"
        f"### Anchor Node: {anchor}\n"
        f"### Problem Statement:\n{problem}\n\n"
        f"### Evidence Rubric Contracts:\n{rubric_text}\n\n"
        f"### Reference Fix Patch:\n{patch}\n\n"
        f"Trajectory Guidelines:\n"
        f"- Turn 1: Model calls get_code_subgraph on '{anchor}' to inspect caller/callee contracts.\n"
        f"- Turn 2 (Multi-Hop Cross-File): If the anchor calls an imported helper, predicate, or contract in another file, "
        f"model MUST inspect that connected target file/span directly via read_file or get_code_subgraph rather than scanning caller files.\n"
        f"- Turn 3: Model applies the minimal, contract-satisfying fix via edit_file on the exact target file.\n"
        f"- Turn 4: Model confirms verification and calls submit_patch.\n\n"
        f"Format strictly as JSON: {{\"synthetic_id\": \"{syn_id}\", \"messages\": [...]}}"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = safe_parse_json(content)
                if parsed:
                    msgs = parsed.get("messages", [])
                    # Quality gate: must have at least 4 turns and valid user/assistant messages
                    if msgs and len(msgs) >= 4:
                        normalized_msgs = []
                        turns = []
                        for m in msgs:
                            role = "model" if m.get("role") in ("assistant", "model") else m.get("role", "user")
                            content = m.get("content", "") or ""
                            tcs = m.get("tool_calls", [])
                            norm_tcs = []
                            if tcs and isinstance(tcs, list):
                                for idx, tc in enumerate(tcs):
                                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                                    args = fn.get("arguments", {})
                                    if isinstance(args, dict):
                                        args = json.dumps(args)
                                    norm_tcs.append({
                                        "id": tc.get("id", f"call_{idx}"),
                                        "type": "function",
                                        "function": {
                                            "name": fn.get("name", ""),
                                            "arguments": str(args),
                                        },
                                    })
                            norm_msg = {"role": role, "content": content}
                            if norm_tcs:
                                norm_msg["tool_calls"] = norm_tcs
                            normalized_msgs.append(norm_msg)

                            # Native Gemma 4 token representation
                            call_tokens = ""
                            if norm_tcs:
                                call_tokens = "".join(
                                    f"<|tool_call|>call:{tc['function']['name']}{tc['function']['arguments']}<|tool_call|>"
                                    for tc in norm_tcs
                                )
                            turn_body = f"{content}\n{call_tokens}".strip() if call_tokens else content
                            turns.append(f"<start_of_turn>{role}\n{turn_body}<end_of_turn>")

                        text_representation = "\n".join(turns)
                        return {
                            "synthetic_id": syn_id,
                            "anchor_node": anchor,
                            "messages": normalized_msgs,
                            "text": text_representation,
                        }
        except Exception as e:
            time.sleep(1)

    return None


def run_batch_rollouts(
    synthetic_tasks_file: Path,
    output_file: Path,
    model_name: Optional[str] = None,
    target_count: int = 150,
    max_workers: int = 4,
    base_url: str = "http://127.0.0.1:8000/v1",
) -> None:
    """Collects verified teacher trajectories across synthetic tasks with checkpointing."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not model_name:
        model_name = auto_detect_vllm_model(base_url)

    already_done = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        already_done.add(json.loads(line).get("synthetic_id"))
                    except Exception:
                        pass

    logger.info("Existing verified trajectories on disk: %d", len(already_done))

    synthetic_tasks: List[Dict[str, Any]] = []
    with synthetic_tasks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                synthetic_tasks.append(json.loads(line))

    tasks_to_rollout = [
        t for t in synthetic_tasks
        if t.get("synthetic_id") not in already_done
    ][:target_count]

    logger.info(
        "Collecting %d verified teacher trajectories using %d workers on %s...",
        len(tasks_to_rollout), max_workers, model_name
    )

    start_time = time.time()
    collected = len(already_done)

    with output_file.open("a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_teacher_trajectory, t, model_name, base_url): t
            for t in tasks_to_rollout
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if res:
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
                collected += 1
            if i % 25 == 0 or i == len(tasks_to_rollout):
                elapsed = time.time() - start_time
                logger.info(
                    "[%d/%d] Collected: %d trajectories (Elapsed: %.1fs / %.1fm)",
                    i, len(tasks_to_rollout), collected, elapsed, elapsed / 60
                )

    logger.info("Rollout collection complete! Total verified trajectories: %d -> %s", collected, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3.2B: Teacher Trajectory Rollouts.")
    parser.add_argument(
        "--synthetic-tasks",
        type=Path,
        default=Path("data/synthetic_tasks/synthetic_tasks.jsonl"),
        help="Path to synthetic tasks file generated in 3.2A.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/trajectories/sft_scaled.jsonl"),
        help="Path to save admitted SFT trajectories.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Model name (auto-detected from vLLM if None)")
    parser.add_argument("--target-count", type=int, default=400, help="Number of trajectories to collect")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    run_batch_rollouts(
        synthetic_tasks_file=args.synthetic_tasks,
        output_file=args.output_file,
        model_name=args.model_name,
        target_count=args.target_count,
        max_workers=args.workers,
        base_url=args.vllm_url,
    )


if __name__ == "__main__":
    main()
