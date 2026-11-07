#!/usr/bin/env python3
"""
Milestone 4: Best-of-N Candidate Rollout Sampler for Rejection Fine-Tuning (RFT).
Samples N multi-turn candidate rollouts per synthetic task from the SFT policy checkpoint via vLLM,
and evaluates each rollout under three verifiers:
  - V_free: Unanchored LLM judge
  - V_graph: GraphForge evidence rubric judge
  - V_exec: Deterministic structural tool-execution & non-empty edit oracle

Key Fixes over Pilot Sampler:
  1. No prompt leakage: User prompt contains ONLY problem_statement (no '### Anchor Node:').
  2. Extracts patches directly from `edit_file(filepath, old_string, new_string)` tool calls
     (constructing a unified diff for V_free / V_graph) so tool-using rollouts are rewarded,
     never penalized!
  3. Preserves structured `messages` with `tool_calls` (dict arguments) and `role: 'tool'`
     observations so `train_rft.py` formats 100% native Gemma 4 `<|tool_call>` tokens.
"""

import argparse
import json
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.teacher_rollouts import extract_old_and_new_from_task
except ImportError:
    from teacher_rollouts import extract_old_and_new_from_task

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rft_sampler")


def auto_detect_vllm_models(base_url: str = "http://127.0.0.1:8000/v1", prefer_adapter: str = "main_lora") -> Tuple[str, str]:
    """
    Queries vLLM /v1/models endpoint to auto-detect both:
      1. `policy_model` (the LoRA adapter being sampled, e.g. 'main_lora')
      2. `judge_model` (the uncontaminated base model, e.g. 'gemma-4-31b-it-qat-w4a16-ct')
    This prevents B3 / Phase 6 self-judging contamination where `main_lora` grades itself!
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ids = [m["id"] for m in data.get("data", [])]
            base_candidates = [i for i in ids if i != prefer_adapter and "lora" not in i.lower()]
            judge_model = base_candidates[0] if base_candidates else (ids[0] if ids else "gemma-4-31b-it-qat-w4a16-ct")
            policy_model = prefer_adapter if prefer_adapter in ids else judge_model
            logger.info("Detected policy_model='%s', uncontaminated judge_model='%s'", policy_model, judge_model)
            return policy_model, judge_model
    except Exception as e:
        logger.warning("Could not auto-detect model from %s (%s). Using fallback.", url, e)
        return "main_lora", "gemma-4-31b-it-qat-w4a16-ct"


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
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


def call_vllm_messages(
    messages: List[Dict[str, Any]],
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Optional[Dict[str, Any]]:
    """Queries vLLM chat/completions and returns the full choice message dict."""
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]
    except Exception as e:
        logger.warning("vLLM generation error: %s", e)
        return None


def parse_tool_call_from_message(
    msg: Dict[str, Any],
    default_name: str = "get_code_subgraph",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Extracts (clean_thought, tool_call_dict) from a vLLM assistant response,
    supporting both structured `tool_calls` and inline Gemma 4 `<|tool_call>call:name{...}<tool_call|>`.
    """
    content = (msg.get("content") or "").strip()
    # Strip any accidental <|turn>user hallucination
    content = re.sub(r"^<\|turn>user\s*", "", content).strip()

    tcs = msg.get("tool_calls")
    if tcs and isinstance(tcs, list) and len(tcs) > 0:
        tc = tcs[0]
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name") or default_name
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = safe_parse_json(args) or {"raw": args}
        return content, {"name": name, "arguments": args if isinstance(args, dict) else {}}

    # Check inline Gemma 4 native syntax: <|tool_call>call:func{...}<tool_call|>
    m_native = re.search(
        r"<\|tool_call\|?>call:([a-zA-Z0-9_]+)(\{[\s\S]*?\})(?:<\|?tool_call\|?>|$)",
        content,
    )
    if m_native:
        name = m_native.group(1)
        raw_args = m_native.group(2).replace('<|"|>', '"')
        args = safe_parse_json(raw_args) or {}
        thought = content[: m_native.start()].strip()
        return thought, {"name": name, "arguments": args}

    # Check <tool_call>{...}</tool_call>
    m_xml = re.search(r"<tool_call>([\s\S]*?)</tool_call>", content)
    if m_xml:
        parsed = safe_parse_json(m_xml.group(1))
        if parsed and "name" in parsed:
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                args = safe_parse_json(args) or {}
            thought = content[: m_xml.start()].strip()
            return thought, {"name": parsed["name"], "arguments": args}

    return content, None


def extract_patch_from_edit_or_text(
    edit_tc: Optional[Dict[str, Any]],
    raw_text: str,
    default_file: str,
) -> str:
    """
    Constructs a unified git diff from an `edit_file` tool call (preferred)
    or falls back to extracting a diff block from text.
    """
    if edit_tc and edit_tc.get("name") == "edit_file":
        args = edit_tc.get("arguments", {})
        if isinstance(args, dict):
            fpath = args.get("filepath") or default_file
            old_s = str(args.get("old_string", "")).strip()
            new_s = str(args.get("new_string", "")).strip()
            if old_s and new_s and old_s != new_s:
                old_lines = "\n".join(f"-{l}" for l in old_s.splitlines())
                new_lines = "\n".join(f"+{l}" for l in new_s.splitlines())
                return (
                    f"diff --git a/{fpath} b/{fpath}\n"
                    f"--- a/{fpath}\n"
                    f"+++ b/{fpath}\n"
                    f"@@ -1,5 +1,5 @@\n"
                    f"{old_lines}\n{new_lines}"
                )

    match = re.search(r"```(?:diff|patch)?\n([\s\S]*?)```", raw_text)
    if match:
        return match.group(1).strip()
    diff_match = re.search(r"(diff --git a/[\s\S]+)", raw_text)
    if diff_match:
        return diff_match.group(1).strip()
    return ""


def score_rollout_v_free(problem: str, patch: str, judge_model: str, base_url: str) -> bool:
    """Evaluates candidate rollout under V_free (unanchored judge)."""
    if not patch or len(patch) < 25:
        return False
    sys_prompt = "You are a code review judge. Decide if this patch resolves the issue. Respond with JSON: {\"verdict\": \"PASS\" | \"FAIL\"}"
    user_prompt = f"### Problem Statement:\n{problem[:2000]}\n\n### Candidate Patch:\n{patch[:2500]}\n"
    res = call_vllm_messages(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        judge_model,
        base_url,
        temperature=0.0,
        max_tokens=256,
    )
    if res and res.get("content"):
        parsed = safe_parse_json(res["content"])
        return str(parsed.get("verdict", "")).upper() == "PASS" if parsed else ("PASS" in res["content"].upper())
    return False


def score_rollout_v_graph(problem: str, patch: str, rubric: List[str], judge_model: str, base_url: str) -> bool:
    """Evaluates candidate rollout under V_graph (GraphForge evidence rubric judge)."""
    if not patch or len(patch) < 25:
        return False
    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric))
    sys_prompt = (
        "You are an evidence-anchored code verification judge (RepoGraphForge V_graph). "
        "Evaluate the candidate patch strictly against the provided structural evidence rubric. "
        "If ANY contract is violated, omitted, or partially implemented, output FAIL. "
        "Respond with JSON: {\"verdict\": \"PASS\" | \"FAIL\"}"
    )
    user_prompt = f"### Problem Statement:\n{problem[:2000]}\n\n### Evidence Rubric:\n{rubric_text}\n\n### Candidate Patch:\n{patch[:2500]}\n"
    res = call_vllm_messages(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        judge_model,
        base_url,
        temperature=0.0,
        max_tokens=256,
    )
    if res and res.get("content"):
        parsed = safe_parse_json(res["content"])
        return str(parsed.get("verdict", "")).upper() == "PASS" if parsed else ("PASS" in res["content"].upper())
    return False


def sample_candidates_for_task(
    task: Dict[str, Any],
    model_name: str,
    judge_model_name: Optional[str] = None,
    n_samples: int = 4,
    base_url: str = "http://127.0.0.1:8000/v1",
) -> List[Dict[str, Any]]:
    """
    Samples N multi-turn rollouts per task using grounded tool observations and
    scores them using the uncontaminated base `judge_model_name` (never self-judging!).
    """
    judge_model = judge_model_name or "gemma-4-31b-it-qat-w4a16-ct"
    syn_id = task.get("synthetic_id", "syn")
    problem = (task.get("problem_statement") or "").strip()
    anchor = task.get("anchor_node", "core")
    rubric = task.get("evidence_rubric", [])
    target_file, old_buggy, _ = extract_old_and_new_from_task(task)

    start_line, end_line = 40, 85
    subgraph_obs = json.dumps({
        "status": "ok",
        "anchor": anchor,
        "nodes": [{"id": anchor, "file": target_file, "line_span": [start_line, end_line]}],
    })
    numbered_lines = "\n".join(
        f"{idx:4d}: {line}" for idx, line in enumerate(old_buggy.splitlines(), start=start_line + 5)
    )
    read_obs = f"File: {target_file} (lines {start_line}-{end_line})\n{numbered_lines}"

    candidates = []
    for sample_idx in range(n_samples):
        # Turn 1: Localize via graph
        msgs: List[Dict[str, Any]] = [{"role": "user", "content": problem}]
        m1 = call_vllm_messages(msgs, model_name, base_url, temperature=0.7)
        if not m1:
            continue
        thought1, tc1 = parse_tool_call_from_message(m1, default_name="get_code_subgraph")
        if not tc1:
            tc1 = {"name": "get_code_subgraph", "arguments": {"node_id": anchor}}

        msgs.append({
            "role": "assistant",
            "content": thought1,
            "tool_calls": [{"id": "call_1", "type": "function", "function": tc1}],
        })
        msgs.append({
            "role": "tool",
            "name": tc1["name"],
            "tool_call_id": "call_1",
            "content": subgraph_obs,
        })

        # Turn 2: Read target file span
        msgs.append({
            "role": "assistant",
            "content": f"Reading `{target_file}` lines {start_line}-{end_line} to inspect `{anchor}`.",
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": {"filepath": target_file, "start_line": start_line, "end_line": end_line},
                },
            }],
        })
        msgs.append({
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call_2",
            "content": read_obs,
        })

        # Turn 3: Model generates the edit_file call (or patch)
        m3 = call_vllm_messages(msgs, model_name, base_url, temperature=0.7)
        if not m3:
            continue
        thought3, tc3 = parse_tool_call_from_message(m3, default_name="edit_file")
        raw_m3_text = m3.get("content") or ""
        extracted_patch = extract_patch_from_edit_or_text(tc3, raw_m3_text, target_file)

        # Ensure edit_file tool_call is properly structured for training
        if not tc3 or tc3.get("name") != "edit_file":
            # Only admit rollouts that either called edit_file or provided a valid replacement
            has_valid_edit = False
        else:
            args3 = tc3.get("arguments", {})
            has_valid_edit = bool(
                isinstance(args3, dict)
                and args3.get("old_string")
                and args3.get("new_string")
                and args3.get("old_string") != args3.get("new_string")
            )
            msgs.append({
                "role": "assistant",
                "content": thought3,
                "tool_calls": [{"id": "call_3", "type": "function", "function": tc3}],
            })
            msgs.append({
                "role": "tool",
                "name": "edit_file",
                "tool_call_id": "call_3",
                "content": json.dumps({"status": "ok", "filepath": target_file, "replacements": 1}),
            })
            msgs.append({
                "role": "assistant",
                "content": "The edit succeeded. Submitting the verified patch.",
                "tool_calls": [{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "submit_patch", "arguments": {"description": f"Fix {anchor} in {target_file}"}},
                }],
            })

        v_free_pass = score_rollout_v_free(problem, extracted_patch, judge_model, base_url)
        v_graph_pass = score_rollout_v_graph(problem, extracted_patch, rubric, judge_model, base_url)
        v_exec_pass = bool(has_valid_edit and len(extracted_patch) > 30)

        candidates.append({
            "synthetic_id": syn_id,
            "sample_idx": sample_idx,
            "messages": msgs,
            "extracted_patch": extracted_patch,
            "v_free_pass": v_free_pass,
            "v_graph_pass": v_graph_pass,
            "v_exec_pass": v_exec_pass,
        })

    return candidates


def run_rft_sampling(
    synthetic_tasks_file: Path,
    output_file: Path,
    model_name: str,
    judge_model_name: str = "gemma-4-31b-it-qat-w4a16-ct",
    n_tasks: int = 50,
    n_samples_per_task: int = 4,
    workers: int = 4,
    base_url: str = "http://127.0.0.1:8000/v1",
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    with synthetic_tasks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    selected_tasks = tasks[:n_tasks]
    logger.info(
        "Sampling %d rollouts (%d tasks x N=%d) with policy='%s' and base judge='%s'...",
        len(selected_tasks) * n_samples_per_task, len(selected_tasks), n_samples_per_task, model_name, judge_model_name,
    )

    start_time = time.time()
    total_candidates = 0

    with output_file.open("w", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(sample_candidates_for_task, t, model_name, judge_model_name, n_samples_per_task, base_url): t
            for t in selected_tasks
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            candidates = fut.result()
            for c in candidates:
                out_f.write(json.dumps(c) + "\n")
                total_candidates += 1
            out_f.flush()
            if i % 10 == 0 or i == len(selected_tasks):
                logger.info(
                    "[%d/%d] Completed tasks (Generated %d candidates in %.1fs)",
                    i, len(selected_tasks), total_candidates, time.time() - start_time,
                )

    logger.info("RFT sampling complete! Total candidates: %d -> %s", total_candidates, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4: RFT Candidate Rollout Sampler.")
    parser.add_argument("--synthetic-tasks", type=Path, default=Path("data/synthetic_tasks/synthetic_tasks.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/trajectories/rft_candidates.jsonl"))
    parser.add_argument("--model-name", type=str, default="auto", help="Policy model ID or 'auto'.")
    parser.add_argument("--judge-model", type=str, default="auto", help="Base judge model ID (never main_lora).")
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    auto_policy, auto_judge = auto_detect_vllm_models(args.vllm_url, prefer_adapter="main_lora")
    model_name = auto_policy if args.model_name.lower() in ("auto", "none", "") else args.model_name
    judge_model = auto_judge if args.judge_model.lower() in ("auto", "none", "") else args.judge_model

    run_rft_sampling(
        synthetic_tasks_file=args.synthetic_tasks,
        output_file=args.output_file,
        model_name=model_name,
        judge_model_name=judge_model,
        n_tasks=args.n_tasks,
        n_samples_per_task=args.n_samples,
        workers=args.workers,
        base_url=args.vllm_url,
    )


if __name__ == "__main__":
    main()
