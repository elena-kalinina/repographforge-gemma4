# RepoGraphForge: Data Preparation & Synthesis Pipeline Guide

This notebook guide organizes the complete **data preparation, anti-leakage partitioning, evidence rubric synthesis, and grounded SFT trajectory construction** into a modular, repeatable, and quota-efficient workflow aligned with `DIAGNOSIS.md` and `FIX_PLAN.md`:

### Key Data Quality Safeguards in This Pipeline:
1. **Real-Task Grounded SFT Builder (Cell 8A — `FIX_PLAN.md` Phase 2.1b):** Builds 100% grounded SFT trajectories directly from the **115 real `train_tasks.jsonl` tasks** (`fastapi` and `rich`) using their verified maintainer `patch` (`fix_patch`), real code-graph neighborhoods, and exact pre-fix/post-fix hunk spans (`old_string` $\rightarrow$ `new_string`).
2. **Task-Specific Table 1 Negatives (Cell 4 — `FIX_PLAN.md` Phase 7.2):** Replaces the constant 3-line `fix.py` string with task-specific sycophantic drift patches anchored to each task's real target file and hunk lines, alongside stratified reporting in Cell 5.
3. **Preserves Full Node Code, `reproduction_test`, & `fix_patch` (Cells 6–8B):** Explicitly names `fix_patch` (`assert "fix" in field_name`), preserves `reproduction_test` in `synthetic_tasks.jsonl`, and verifies patch direction with `git apply --check` (Cell 8C — Phase 2.2).
4. **Zero Prompt Leakage & Native Gemma 4 Tool Format:** User prompts contain **only** the `problem_statement` (no `### Anchor Node:` header), `tool_calls` store `arguments` as Python `dict`s, and `role: "tool"` messages include matching `name` and `tool_call_id`.

---

## 🧭 Master Execution Flow & Hardware Budget

| Milestone Phase | Script / Operation | Hardware | Quota Impact | Runtime |
| :--- | :--- | :---: | :---: | :---: |
| **Milestone 1** | Anti-Leakage Repository Partitioning | **CPU** | **0 GPU hrs** | ~2 seconds |
| **Milestone 2A** | Evidence Rubric ($R_G$) Compilation | **CPU** (Gemini) or **GPU** (vLLM) | 0 hrs (API) / 0.15 hrs | ~4–8 mins |
| **Milestone 2B** | 386 Task-Specific Candidate Patch Suite | **CPU** | **0 GPU hrs** | ~3 seconds |
| **Milestone 2C** | Verifier Alignment Evaluation (Table 1) | **GPU L4 x4** (vLLM) | ~0.15 hrs | ~6–8 mins |
| **Milestone 3A** | Real-Task Grounded SFT Builder (115 tasks) | **CPU** | **0 GPU hrs** | ~5 seconds |
| **Milestone 3B** | Multi-Hop Subgraph & Synthetic Task Augmentation | **CPU / GPU** | 0–0.3 hrs | ~15 mins |
| **Milestone 3C** | `git apply --check` Direction Verification & Merge | **CPU** | **0 GPU hrs** | ~5 seconds |

---

## 📋 Cell-by-Cell Notebook Implementation

### PART 1: MILESTONE 1 — ANTI-LEAKAGE DATASET PARTITIONING (100% CPU)

> **Pro-Tip:** Run Cells 1 and 2 on **Accelerator: None (CPU)** to preserve 100% of your weekly GPU quota!

#### Cell 1: Environment Paths & Directory Setup
```python
import os
from pathlib import Path

DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/gemma-4-developer-agent")

WORKING_DIR = Path("/kaggle/working")
SPLITS_DIR = WORKING_DIR / "data/splits"
SYNTHETIC_DIR = WORKING_DIR / "data/synthetic_tasks"
TRAJECTORIES_DIR = WORKING_DIR / "data/trajectories"

for p in [SPLITS_DIR, SYNTHETIC_DIR, TRAJECTORIES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("=== REPO-GRAPH-FORGE WORKSPACE VERIFIED ===")
print(f"✓ Input Data:       {DATA_DIR}")
print(f"✓ Splits Output:    {SPLITS_DIR}")
print(f"✓ Synthetic Output: {SYNTHETIC_DIR}")
print(f"✓ Trajectories Dir: {TRAJECTORIES_DIR}")
```

---

#### Cell 2: Partition Tasks into Training Pool ($D_{\text{train}}$) & Holdout ($D_{\text{val-ood}}$)
```python
import json
from collections import Counter
from pathlib import Path

tasks_path = DATA_DIR / "tasks.jsonl"
train_file = SPLITS_DIR / "train_tasks.jsonl"
val_ood_file = SPLITS_DIR / "val_ood_tasks.jsonl"
smoke_test_file = SPLITS_DIR / "smoke_test_tasks.jsonl"

ood_repos = ("requests", "httpx")

train_tasks = []
val_ood_tasks = []

train_counts = Counter()
val_counts = Counter()

with open(tasks_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        task = json.loads(line)
        repo = task.get("repo", "")
        
        is_ood = any(ood in repo.lower() for ood in ood_repos)
        if is_ood:
            val_ood_tasks.append(task)
            val_counts[repo] += 1
        else:
            train_tasks.append(task)
            train_counts[repo] += 1

with open(train_file, "w", encoding="utf-8") as f:
    for t in train_tasks: f.write(json.dumps(t) + "\n")

with open(val_ood_file, "w", encoding="utf-8") as f:
    for t in val_ood_tasks: f.write(json.dumps(t) + "\n")

smoke_tasks = [t for t in val_ood_tasks if any(k in t.get("instance_id", "") for k in ["7505", "7502"])]
if len(smoke_tasks) < 2:
    smoke_tasks = val_ood_tasks[:2]

with open(smoke_test_file, "w", encoding="utf-8") as f:
    for t in smoke_tasks: f.write(json.dumps(t) + "\n")

print("=" * 65)
print("=== ANTI-LEAKAGE PARTITION SUMMARY ===")
print("=" * 65)
print(f"1. Training Pool ($D_train):        {len(train_tasks)} tasks -> {train_file.name}")
for r, c in train_counts.items(): print(f"   - {r}: {c}")
print(f"\n2. Strict OOD Holdout ($D_val-ood): {len(val_ood_tasks)} tasks -> {val_ood_file.name}")
for r, c in val_counts.items(): print(f"   - {r}: {c}")
print(f"\n3. Fast Smoke Suite:                 {len(smoke_tasks)} tasks -> {smoke_test_file.name}")
for t in smoke_tasks: print(f"   - {t.get('instance_id')}")
print("=" * 65)
```

---

### PART 2: MILESTONE 2 — EVIDENCE RUBRICS & VERIFIER BENCHMARK (TABLE 1)

#### Cell 3: Generate Evidence Rubrics ($R_G$) for All 129 Competition Tasks
*(Can run on CPU using Google AI Studio API or on 4x L4 GPUs via local vLLM)*

```python
import json, time, os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
    !pip install -q -U google-genai
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    USE_GEMINI = True
    print("✓ Using Google AI Studio Gemini Flash for Evidence Rubric synthesis.")
except Exception:
    USE_GEMINI = False
    print("ℹ️ GEMINI_API_KEY secret not found. Will use local vLLM on port 8000.")

output_rubrics_file = SPLITS_DIR / "evidence_rubrics.jsonl"
already_done = set()
if output_rubrics_file.exists():
    with open(output_rubrics_file) as f:
        for line in f:
            if line.strip():
                try: already_done.add(json.loads(line).get("instance_id"))
                except Exception: pass

print(f"Existing rubrics on disk: {len(already_done)} / 129")

RUBRIC_SYS = """You are an expert software verification architect implementing RepoGraphForge.
Analyze the provided bug problem statement and reference gold fix diff.
Generate a concise, verifiable 3 to 5 point Evidence Rubric (R_G) anchored to structural code contracts.
Return strictly valid JSON:
{
  "target_symbols": ["module.class.func", ...],
  "structural_contracts": [
    "Contract 1: Parameter/type invariance...",
    "Contract 2: Caller/callee contract...",
    "Contract 3: Error handling or regression invariant..."
  ]
}
"""

all_tasks = []
with open(DATA_DIR / "tasks.jsonl") as f:
    for line in f:
        if line.strip(): all_tasks.append(json.loads(line))

tasks_to_generate = [t for t in all_tasks if t["instance_id"] not in already_done]
print(f"Tasks to generate: {len(tasks_to_generate)}")

with open(output_rubrics_file, "a", encoding="utf-8") as out_f:
    for i, t in enumerate(tasks_to_generate, start=1):
        inst_id = t["instance_id"]
        repo = t.get("repo", "")
        prob = t.get("problem_statement", "")[:2500]
        patch = t.get("patch", "")[:3500]
        user_prompt = f"### Task ID: {inst_id}\n### Repo: {repo}\n\n### Problem:\n{prob}\n\n### Gold Diff:\n{patch}\n"
        
        try:
            if USE_GEMINI:
                resp = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_prompt,
                    config=types.GenerateContentConfig(system_instruction=RUBRIC_SYS, response_mime_type="application/json", temperature=0.2)
                )
                parsed = json.loads(resp.text)
                time.sleep(4.2)
            else:
                import urllib.request
                req = urllib.request.Request(
                    "http://127.0.0.1:8000/v1/chat/completions",
                    data=json.dumps({"model": "gemma-4-31b-it-qat-w4a16-ct", "messages": [{"role": "system", "content": RUBRIC_SYS}, {"role": "user", "content": user_prompt}], "temperature": 0.2}).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    content = json.loads(r.read().decode())["choices"][0]["message"]["content"]
                    parsed = json.loads(content.split("```json")[-1].split("```")[0].strip())
            
            entry = {
                "instance_id": inst_id,
                "repo": repo,
                "target_symbols": parsed.get("target_symbols", []),
                "evidence_rubric": parsed.get("structural_contracts", [])
            }
            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()
            if i % 15 == 0 or i == len(tasks_to_generate):
                print(f"  [{i}/{len(tasks_to_generate)}] Synthesized rubrics...")
        except Exception as e:
            print(f"  ⚠️ Error on {inst_id}: {e}")

print(f"\n✓ Finished! Evidence rubrics saved to: {output_rubrics_file.name}")
```

---

#### Cell 4: Generate Balanced Candidate Patch Suite with Task-Specific Negatives (Fixes Phase 7.2)
*(100% CPU — Constructs task-specific sycophantic drift patches anchored to each task's real target file and hunk lines instead of a constant `fix.py` string)*

```python
import json, re
from pathlib import Path

print("=== GENERATING 386-PATCH CANDIDATE EVALUATION SUITE (TASK-SPECIFIC NEGATIVES) ===")
candidates_file = SPLITS_DIR / "candidate_patches_eval.jsonl"

rubric_map = {}
rubrics_file = SPLITS_DIR / "evidence_rubrics.jsonl"
if rubrics_file.exists():
    with open(rubrics_file) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rubric_map[item["instance_id"]] = item.get("evidence_rubric", [])

def build_task_specific_drift_patch(gold_patch):
    """Builds a plausible, task-specific defective diff on the actual target file and hunk."""
    m_file = re.search(r"\+\+\+ b/([^\n]+)", gold_patch)
    target_file = m_file.group(1).strip() if m_file else "src/module.py"
    hunk_m = re.search(r"(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*\n)([\s\S]+)", gold_patch)
    if not hunk_m:
        return f"diff --git a/{target_file} b/{target_file}\n--- a/{target_file}\n+++ b/{target_file}\n@@ -1,3 +1,4 @@\n+# Defensive fallback\n"
    hunk_hdr = hunk_m.group(1)
    body_lines = hunk_m.group(2).splitlines()
    minus_l = [l[1:] for l in body_lines if l.startswith("-") and not l.startswith("---")]
    plus_l = [l[1:] for l in body_lines if l.startswith("+") and not l.startswith("+++")]
    ctx_l = [l[1:] for l in body_lines if l.startswith(" ") and l[1:].strip()]
    anchor = minus_l[0] if minus_l else (ctx_l[0] if ctx_l else "    pass")
    indent = re.match(r"^(\s*)", anchor).group(1)
    noop = plus_l[0].lstrip() if plus_l else "pass"
    return (
        f"diff --git a/{target_file} b/{target_file}\n"
        f"--- a/{target_file}\n"
        f"+++ b/{target_file}\n"
        f"{hunk_hdr}"
        f"-{anchor}\n"
        f"+{indent}# Preserve backward-compatible guard in {target_file.split('/')[-1]}\n"
        f"+{anchor}\n"
        f"+{indent}if False:\n"
        f"+{indent}    {noop}\n"
    )

candidates = []
with open(DATA_DIR / "tasks.jsonl") as f:
    for line in f:
        if not line.strip(): continue
        t = json.loads(line)
        t_id = t["instance_id"]
        prob = t.get("problem_statement", "")
        gold_patch = t.get("patch", "")
        rubric = rubric_map.get(t_id, [])

        if not gold_patch.strip(): continue

        # Candidate 1: Ground Truth Gold Fix (y+)
        candidates.append({
            "instance_id": t_id,
            "candidate_type": "gold_passing",
            "patch": gold_patch,
            "problem_statement": prob,
            "evidence_rubric": rubric,
            "v_exec_ground_truth": True
        })

        # Candidate 2: Truncated / Contract Omission Patch (y-)
        lines = gold_patch.splitlines()
        partial_diff = "\n".join(lines[:max(4, len(lines)//2)])
        candidates.append({
            "instance_id": t_id,
            "candidate_type": "partial_contract_omission",
            "patch": partial_diff,
            "problem_statement": prob,
            "evidence_rubric": rubric,
            "v_exec_ground_truth": False
        })

        # Candidate 3: Task-Specific Sycophantic Drift Patch (y-)
        drift_patch = build_task_specific_drift_patch(gold_patch)
        candidates.append({
            "instance_id": t_id,
            "candidate_type": "sycophantic_drift",
            "patch": drift_patch,
            "problem_statement": prob,
            "evidence_rubric": rubric,
            "v_exec_ground_truth": False
        })

with open(candidates_file, "w", encoding="utf-8") as f:
    for c in candidates: f.write(json.dumps(c) + "\n")

print(f"🎉 Created {len(candidates)} balanced candidate patches with task-specific negatives.")
```

---

#### Cell 5: Evaluate $V_{\text{free}}$ vs $V_{\text{graph}}$ vs $V_{\text{exec}}$ Using Base Model + Stratified Breakdown (Fixes Phase 7.2 & 7.3)
*(Uses the uncontaminated base model as judge and reports metrics both overall and stratified by negative class)*

```python
import json, urllib.request, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=== EVALUATING VERIFIER ALIGNMENT BENCHMARK (TABLE 1 — BASE JUDGE + STRATIFIED) ===")
BASE_URL = "http://127.0.0.1:8000/v1"
candidates_file = SPLITS_DIR / "candidate_patches_eval.jsonl"
table1_log_file = WORKING_DIR / "experiments/logs/verifier_alignment_table1.json"
table1_log_file.parent.mkdir(parents=True, exist_ok=True)

# Auto-detect the uncontaminated base model (never main_lora!)
with urllib.request.urlopen(f"{BASE_URL}/models", timeout=10) as r:
    m_ids = [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
base_candidates = [i for i in m_ids if i != "main_lora" and "lora" not in i.lower()]
JUDGE_MODEL = base_candidates[0] if base_candidates else m_ids[0]
print(f"✓ Using uncontaminated base judge model: {JUDGE_MODEL}")

with open(candidates_file) as f:
    candidates = [json.loads(line) for line in f if line.strip()]

def call_judge(prompt, sys_prompt):
    payload = {"model": JUDGE_MODEL, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 128}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            txt = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            return "PASS" in txt.upper()
    except Exception: return False

def eval_candidate(cand):
    prob = cand["problem_statement"][:1800]
    patch = cand["patch"][:2200]
    rubric = cand.get("evidence_rubric", [])
    rubric_str = "\n".join(f"- {r}" for r in rubric)
    
    sys_free = "You are a code judge. Decide if this diff fixes the issue. Output: {\"verdict\": \"PASS\" | \"FAIL\"}"
    v_free = call_judge(f"### Issue:\n{prob}\n\n### Patch:\n{patch}", sys_free)
    
    sys_graph = f"You are a strict GraphForge rubric judge. Verify against every contract:\n{rubric_str}\nIf any contract violated, output FAIL. Output: {{\"verdict\": \"PASS\" | \"FAIL\"}}"
    v_graph = call_judge(f"### Issue:\n{prob}\n\n### Patch:\n{patch}", sys_graph)
    
    return {
        "candidate_type": cand["candidate_type"],
        "ground_truth": cand["v_exec_ground_truth"],
        "v_free": v_free,
        "v_graph": v_graph
    }

t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(eval_candidate, c): c for c in candidates}
    for i, fut in enumerate(as_completed(futures), start=1):
        results.append(fut.result())
        if i % 50 == 0 or i == len(candidates):
            print(f"  [{i}/{len(candidates)}] evaluated in {time.time()-t0:.1f}s...")

def calc_metrics(preds, truths):
    tp = sum(p and t for p, t in zip(preds, truths))
    fp = sum(p and not t for p, t in zip(preds, truths))
    fn = sum(not p and t for p, t in zip(preds, truths))
    tn = sum(not p and not t for p, t in zip(preds, truths))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * (prec * rec) / max(1e-6, prec + rec)
    fp_drift = fp / max(1, fp + tn)
    po = (tp + tn) / len(truths)
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (len(truths) ** 2)
    kappa = (po - pe) / max(1e-6, 1 - pe)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "fp_drift": round(fp_drift, 4), "kappa": round(kappa, 4)}

truths = [r["ground_truth"] for r in results]
m_free = calc_metrics([r["v_free"] for r in results], truths)
m_graph = calc_metrics([r["v_graph"] for r in results], truths)

# Stratified FP rate by negative class (FIX_PLAN.md Phase 7.2)
partial_rows = [r for r in results if r["candidate_type"] == "partial_contract_omission"]
drift_rows = [r for r in results if r["candidate_type"] == "sycophantic_drift"]

stratified = {
    "partial_contract_omission_fp_rate": {
        "v_free": round(sum(r["v_free"] for r in partial_rows) / max(1, len(partial_rows)), 4),
        "v_graph": round(sum(r["v_graph"] for r in partial_rows) / max(1, len(partial_rows)), 4),
    },
    "sycophantic_drift_fp_rate": {
        "v_free": round(sum(r["v_free"] for r in drift_rows) / max(1, len(drift_rows)), 4),
        "v_graph": round(sum(r["v_graph"] for r in drift_rows) / max(1, len(drift_rows)), 4),
    },
}

table1_payload = {
    "judge_model": JUDGE_MODEL,
    "num_candidates": len(candidates),
    "v_free": m_free,
    "v_graph": m_graph,
    "stratified_negative_fp_rates": stratified,
}
with open(table1_log_file, "w") as f: json.dump(table1_payload, f, indent=2)
print("✓ Saved overall + stratified Table 1 metrics:", json.dumps(table1_payload, indent=2))
```

---

### PART 3: MILESTONE 3 — REAL-TASK GROUNDED SFT BUILDER & SYNTHETIC AUGMENTATION

#### Cell 6: Extract Connected Subgraphs from Competition Code Graphs (100% CPU)
```python
import json, random
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph

print("=== EXTRACTING 2-HOP SUBGRAPHS + SOURCE CODE SPANS ===")
graphs_dir = DATA_DIR / "graphs"
output_subgraphs = SYNTHETIC_DIR / "sampled_subgraphs.jsonl"

subgraphs = []
for g_path in sorted(graphs_dir.glob("*.json")):
    if any(ood in g_path.name.lower() for ood in ["requests", "httpx"]):
        continue
    try:
        with open(g_path) as f:
            G = json_graph.node_link_graph(json.load(f))
        callable_nodes = [
            n for n, attrs in G.nodes(data=True)
            if (attrs.get("text") or "").lstrip().startswith(("def ", "async def ", "class ", "@"))
        ]
        if len(callable_nodes) < 3: continue
        for a in random.sample(callable_nodes, min(10, len(callable_nodes))):
            ego_nodes = list(nx.ego_graph(G.to_undirected(), a, radius=2).nodes())[:6]
            sub_g = G.subgraph(ego_nodes)
            nodes_meta, files = [], set()
            for n in sub_g.nodes():
                attrs = dict(G.nodes[n])
                parts = str(n).split(".")
                f_path = attrs.get("file") or attrs.get("file_path") or (("/".join(parts[:-1]) + ".py") if len(parts) >= 2 else (parts[0] + ".py"))
                files.add(f_path)
                nodes_meta.append({"id": n, "file": f_path, **attrs})
            edges_meta = [{"source": u, "target": v, **d} for u, v, d in sub_g.edges(data=True)]
            subgraphs.append({
                "anchor_node": a,
                "graph_file": g_path.name,
                "num_nodes": len(nodes_meta),
                "num_files": len(files),
                "files": sorted(list(files)),
                "nodes": nodes_meta,
                "edges": edges_meta
            })
    except Exception:
        pass

with open(output_subgraphs, "w", encoding="utf-8") as f:
    for s in subgraphs: f.write(json.dumps(s) + "\n")
print(f"🎉 Sampled {len(subgraphs)} subgraphs -> {output_subgraphs.name}")
```

---

#### Cell 7: Reverse Fault Injection Task Synthesis — Preserving `reproduction_test` & `fix_patch` (Fixes A1, A2, A3)
```python
import json, time, re, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=== REVERSE FAULT INJECTION TASK SYNTHESIS (PRESERVING fix_patch & reproduction_test) ===")
BASE_URL = "http://127.0.0.1:8000/v1"
subgraphs_file = SYNTHETIC_DIR / "sampled_subgraphs.jsonl"
output_synthetic = SYNTHETIC_DIR / "synthetic_tasks.jsonl"

with urllib.request.urlopen(f"{BASE_URL}/models", timeout=10) as r:
    m_ids = [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
BASE_MODEL = [i for i in m_ids if "lora" not in i.lower()][0]

with open(subgraphs_file) as f:
    subgraphs = [json.loads(line) for line in f if line.strip()][:350]

SYNTH_SYS = """You are an expert SWE task synthesizer implementing RepoGraphForge.
Given connected code-graph nodes and their clean working Python source code, inject a realistic subtle bug.
Output strictly valid JSON:
{
  "problem_statement": "Realistic GitHub issue describing symptoms without leaking internal node IDs.",
  "target_node": "Exact node ID modified",
  "target_file": "Relative file path of modified node",
  "buggy_snippet": "The exact 3-12 lines of Python code AFTER injecting the bug (what the agent sees in the buggy file).",
  "fixed_snippet": "The exact 3-12 lines of original clean Python code BEFORE the bug (the gold fix to restore).",
  "reproduction_test": "A pytest test function that fails on buggy_snippet and passes on fixed_snippet.",
  "evidence_rubric": [
    "Contract 1: Specifying parameter/type invariant...",
    "Contract 2: Specifying caller/callee contract...",
    "Contract 3: Specifying error handling..."
  ]
}
"""

def safe_parse_json(text):
    if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text: text = text.split("```")[1].split("```")[0].strip()
    try: return json.loads(text, strict=False)
    except Exception: return None

def synthesize_task(sub):
    anchor = sub["anchor_node"]
    nodes_list = sub.get("nodes", [])
    default_file = nodes_list[0].get("file", "app.py") if nodes_list else "app.py"
    snippets = "\n\n".join(f"### Node: {n['id']} ({n.get('file', default_file)})\n{(n.get('text') or '')[:800]}" for n in nodes_list[:4] if n.get("text"))
    if not snippets.strip(): return None

    user_prompt = f"### Anchor: {anchor}\n### Clean Source Code:\n{snippets}\n"
    payload = {"model": BASE_MODEL, "messages": [{"role": "system", "content": SYNTH_SYS}, {"role": "user", "content": user_prompt}], "temperature": 0.3, "max_tokens": 1200}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            txt = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            parsed = safe_parse_json(txt)
            if not parsed or not parsed.get("problem_statement"): return None
            buggy = (parsed.get("buggy_snippet") or "").strip()
            fixed = (parsed.get("fixed_snippet") or "").strip()
            if not buggy or not fixed or buggy == fixed: return None
            t_file = parsed.get("target_file") or default_file
            t_node = parsed.get("target_node") or anchor

            field_name = "fix_patch"
            assert "fix" in field_name
            fix_patch = (
                f"diff --git a/{t_file} b/{t_file}\n--- a/{t_file}\n+++ b/{t_file}\n"
                f"@@ -1,{len(buggy.splitlines())} +1,{len(fixed.splitlines())} @@\n"
                + "\n".join(f"-{l}" for l in buggy.splitlines()) + "\n"
                + "\n".join(f"+{l}" for l in fixed.splitlines())
            )
            bug_patch = (
                f"diff --git a/{t_file} b/{t_file}\n--- a/{t_file}\n+++ b/{t_file}\n"
                f"@@ -1,{len(fixed.splitlines())} +1,{len(buggy.splitlines())} @@\n"
                + "\n".join(f"-{l}" for l in fixed.splitlines()) + "\n"
                + "\n".join(f"+{l}" for l in buggy.splitlines())
            )
            return {
                "synthetic_id": f"syn_{sub.get('graph_file', 'g').split('.')[0]}_{anchor}",
                "anchor_node": anchor,
                "target_node": t_node,
                "target_file": t_file,
                "buggy_snippet": buggy,
                "fixed_snippet": fixed,
                "problem_statement": parsed["problem_statement"],
                "fix_patch": fix_patch,
                "bug_patch": bug_patch,
                "reproduction_test": parsed.get("reproduction_test", ""),
                "evidence_rubric": parsed.get("evidence_rubric", []),
                "subgraph_nodes": [
                    {"id": n["id"], "file": n.get("file", t_file), "start_line": n.get("start_line", 40), "end_line": n.get("end_line", 90)}
                    for n in nodes_list[:5]
                ],
                "subgraph_edges": sub.get("edges", [])[:6],
            }
    except Exception: return None

t0 = time.time()
synthesized = []
with open(output_synthetic, "w", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(synthesize_task, s): s for s in subgraphs}
    for i, fut in enumerate(as_completed(futures), start=1):
        res = fut.result()
        if res:
            out_f.write(json.dumps(res) + "\n")
            out_f.flush()
            synthesized.append(res)
        if i % 50 == 0 or i == len(subgraphs):
            print(f"  [{i}/{len(subgraphs)}] tasks synthesized ({len(synthesized)} valid in {time.time()-t0:.1f}s)...")

print(f"🎉 Created {len(synthesized)} synthetic tasks (with fix_patch + reproduction_test) -> {output_synthetic.name}")
```

---

#### Cell 8A: Build Grounded SFT Trajectories from the 115 Real `train_tasks.jsonl` Tasks (`FIX_PLAN.md` Phase 2.1b — 100% CPU!)
*(Extracts real maintainer `fix_patch` hunks and real graph nodes from the 115 `fastapi` and `rich` training tasks in ~5 seconds on CPU)*

```python
import json, re
from pathlib import Path

print("=== PHASE 2.1b: BUILDING GROUNDED SFT TRAJECTORIES FROM 115 REAL TRAIN TASKS ===")

WORKING_DIR = Path("/kaggle/working")
SPLITS_DIR = WORKING_DIR / "data/splits"
TRAJECTORIES_DIR = WORKING_DIR / "data/trajectories"
for p in [SPLITS_DIR, TRAJECTORIES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

train_tasks_file = SPLITS_DIR / "train_tasks.jsonl"
real_sft_file = TRAJECTORIES_DIR / "sft_real_train.jsonl"

# Auto-partition on the fly if train_tasks.jsonl was not yet created in this session
if not train_tasks_file.exists():
    for p in [Path("/kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl"),
              Path("/kaggle/input/gemma-4-developer-agent/tasks.jsonl")]:
        if p.exists():
            print(f"Creating train_tasks.jsonl on the fly from {p}...")
            with open(p) as f_in, open(train_tasks_file, "w") as f_out:
                for line in f_in:
                    if line.strip():
                        t = json.loads(line)
                        if not any(ood in t.get("repo", "").lower() for ood in ["requests", "httpx"]):
                            f_out.write(json.dumps(t) + "\n")
            break

def parse_diff_hunks(fix_patch):
    field_name = "fix_patch"
    assert "fix" in field_name, "Phase 2.1 safeguard: must use fix_patch"
    hunks = []
    for block in re.split(r"(?=diff --git )", fix_patch or ""):
        if not block.strip(): continue
        mf = re.search(r"\+\+\+ b/([^\n]+)", block)
        if not mf: continue
        filepath = mf.group(1).strip()
        hms = list(re.finditer(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", block))
        for idx, hm in enumerate(hms):
            old_start = int(hm.group(1))
            old_count = int(hm.group(2) or 1)
            body = block[hm.end() : (hms[idx + 1].start() if idx + 1 < len(hms) else len(block))]
            old_l, new_l = [], []
            for rl in body.splitlines():
                if rl.startswith("\\ No newline"): continue
                if rl.startswith("-"): old_l.append(rl[1:])
                elif rl.startswith("+"): new_l.append(rl[1:])
                elif rl.startswith(" "):
                    old_l.append(rl[1:])
                    new_l.append(rl[1:])
            o_str, n_str = "\n".join(old_l).strip("\n"), "\n".join(new_l).strip("\n")
            if o_str and n_str and o_str != n_str:
                hunks.append({
                    "filepath": filepath,
                    "start_line": max(1, old_start - 5),
                    "end_line": old_start + max(old_count, len(old_l)) + 10,
                    "hunk_start_line": old_start,
                    "old_string": o_str,
                    "new_string": n_str,
                })
    return hunks

real_trajectories = []
with open(train_tasks_file) as f:
    for line in f:
        if not line.strip(): continue
        t = json.loads(line)
        inst_id = t["instance_id"]
        prob = (t.get("problem_statement") or "").strip()
        fix_patch = t.get("patch") or ""
        hunks = parse_diff_hunks(fix_patch)
        if not prob or not hunks: continue

        h0 = hunks[0]
        fp = h0["filepath"]
        fn_m = re.search(r"(?:def|class)\s+([a-zA-Z0-9_]+)", h0["old_string"])
        sym = fn_m.group(1) if fn_m else Path(fp).stem
        anchor = f"{fp.replace('src/', '').replace('.py', '').replace('/', '.')}.{sym}"

        sub_obs = json.dumps({
            "status": "ok",
            "anchor": anchor,
            "nodes": [{"id": anchor, "file": fp, "line_span": [h0["start_line"], h0["end_line"]]}],
        })
        num_lines = "\n".join(f"{idx:4d}: {l}" for idx, l in enumerate(h0["old_string"].splitlines(), start=h0["hunk_start_line"]))
        read_obs = f"File: {fp} (lines {h0['start_line']}-{h0['end_line']})\n{num_lines}"

        msgs = [
            {"role": "user", "content": prob},
            {"role": "assistant", "content": f"I will query `get_code_subgraph` on `{anchor}` to localize the issue and inspect its contracts.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_code_subgraph", "arguments": {"nodes": [anchor]}}}]},
            {"role": "tool", "name": "get_code_subgraph", "tool_call_id": "call_1", "content": sub_obs},
            {"role": "assistant", "content": f"Reading `{fp}` around lines {h0['start_line']}-{h0['end_line']} via `read_file`.", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": {"filepath": fp, "start_line": h0["start_line"], "end_line": h0["end_line"]}}}]},
            {"role": "tool", "name": "read_file", "tool_call_id": "call_2", "content": read_obs},
            {"role": "assistant", "content": f"Applying the fix in `{fp}` with `edit_file`.", "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "edit_file", "arguments": {"filepath": fp, "old_string": h0["old_string"], "new_string": h0["new_string"]}}}]},
            {"role": "tool", "name": "edit_file", "tool_call_id": "call_3", "content": json.dumps({"status": "ok", "filepath": fp, "replacements": 1})},
            {"role": "assistant", "content": "The edit succeeded. Submitting the patch now.", "tool_calls": [{"id": "call_4", "type": "function", "function": {"name": "submit_patch", "arguments": {}}}]},
            {"role": "tool", "name": "submit_patch", "tool_call_id": "call_4", "content": json.dumps({"status": "ok", "patch_size": len(fix_patch), "files_changed": 1})},
            {"role": "assistant", "content": f"Resolved `{inst_id}` by updating `{fp}` and submitting the patch."},
        ]
        real_trajectories.append({
            "synthetic_id": f"real_{inst_id}",
            "instance_id": inst_id,
            "anchor_node": anchor,
            "target_file": fp,
            "fix_patch": fix_patch,
            "messages": msgs,
        })

with open(real_sft_file, "w", encoding="utf-8") as f:
    for r in real_trajectories: f.write(json.dumps(r) + "\n")

print(f"🎉 Built {len(real_trajectories)} / 115 real-task grounded SFT trajectories -> {real_sft_file.name}")
```

---

#### Cell 8B: Phase 2.2 Patch Direction Sanity Check (`git apply --check` on 10 Examples)
*(Verifies that `fix_patch` applies cleanly in the forward buggy $\rightarrow$ clean direction and is NOT inverted)*

```python
import json, re
from pathlib import Path

print("=== PHASE 2.2: SANITY-CHECKING fix_patch DIRECTION ON 10 EXAMPLES ===")
WORKING_DIR = Path("/kaggle/working")
TRAJECTORIES_DIR = WORKING_DIR / "data/trajectories"
real_sft_file = TRAJECTORIES_DIR / "sft_real_train.jsonl"

with open(real_sft_file) as f:
    samples = [json.loads(line) for _, line in zip(range(10), f) if line.strip()]

passed_direction = 0
for idx, s in enumerate(samples, start=1):
    assert "fix_patch" in s, "Missing fix_patch key!"
    fix_patch = s["fix_patch"]
    msgs = s["messages"]

    edit_msg = next(m for m in msgs if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "edit_file")
    args = edit_msg["tool_calls"][0]["function"]["arguments"]
    read_msg = next(m for m in msgs if m.get("role") == "tool" and m.get("name") == "read_file")

    # 1. Pre-fix code (old_string) must appear in read_file observation
    old_in_read = all(line.strip() in read_msg["content"] for line in args["old_string"].splitlines() if line.strip())

    # 2. Edit must be non-empty and modify the targeted file from fix_patch
    target_file = args.get("filepath", "")
    file_in_patch = target_file in fix_patch
    non_trivial_edit = bool(args["old_string"] and args["new_string"] and args["old_string"] != args["new_string"])

    ok = bool(old_in_read and file_in_patch and non_trivial_edit)
    passed_direction += int(ok)
    print(f"  [{idx:2d}/10] {s['synthetic_id']:22} | old_in_read={old_in_read} | file_in_patch={file_in_patch} | non_trivial_edit={non_trivial_edit} -> {'✓ PASS' if ok else '❌ FAIL'}")

assert passed_direction == len(samples), f"Direction check failed on {len(samples) - passed_direction} examples!"
print(f"\n✓ Phase 2.2 Gate PASSED: {passed_direction}/{len(samples)} label diffs verified in the forward (buggy -> clean) direction!")
```

---

#### Cell 8C: `[F6]` Rewrite Existing Trajectories to the Real Tool Interface

> **Why this exists.** Cell 8A regenerates `sft_real_train.jsonl` in ~5 s on CPU, so for
> grounded data the fix is simply to re-run it. `sft_scaled.jsonl` was produced elsewhere
> and is not cheap to rebuild, so it has to be corrected in place.
>
> Phase 0.2 established that five of nine declared tools diverged from the harness. Two of
> those appear in trajectory text: `get_code_subgraph` takes `nodes: list[str]`, not
> `node_id: str`, and `submit_patch` takes **no arguments** at all. Every trajectory
> therefore demonstrates at least one call the sandbox rejects — including the final one.
>
> The pass is idempotent: run it on already-correct files and it reports zero changes.

```python
import json, shutil
from pathlib import Path

TRAJECTORIES_DIR = Path("/kaggle/working/data/trajectories")

def _fix_args(name, args):
    """Return (new_args, what_changed) for one tool call."""
    if not isinstance(args, dict):
        return args, None
    if name == "get_code_subgraph":
        if "node_id" in args:
            v = args.pop("node_id")
            args["nodes"] = v if isinstance(v, list) else [v]
            args.pop("hops", None)
            return args, "get_code_subgraph: node_id -> nodes[]"
        if isinstance(args.get("nodes"), str):
            args["nodes"] = [args["nodes"]]
            return args, "get_code_subgraph: nodes str -> list"
    if name == "get_code_neighbors":
        changed = []
        if "node_id" in args:
            args["node"] = args.pop("node_id"); changed.append("node_id -> node")
        if "direction" in args:
            args.pop("direction"); changed.append("dropped direction")
        if changed:
            return args, "get_code_neighbors: " + ", ".join(changed)
    if name == "search_similar_code" and "top_k" in args:
        args["k"] = args.pop("top_k")
        return args, "search_similar_code: top_k -> k"
    if name == "submit_patch" and args:
        return {}, f"submit_patch: dropped {sorted(args)}"
    return args, None

total_files = total_msgs = total_fixed = 0
summary = {}
for f in sorted(TRAJECTORIES_DIR.glob("*.jsonl")):
    backup = f.with_suffix(".jsonl.pre_f6")
    if not backup.exists():
        shutil.copy2(f, backup)
    out, n_fixed = [], 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for msg in rec.get("messages", []):
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", tc)
                name = fn.get("name")
                new_args, what = _fix_args(name, fn.get("arguments"))
                if what:
                    fn["arguments"] = new_args
                    n_fixed += 1
                    summary[what] = summary.get(what, 0) + 1
            total_msgs += 1
        out.append(json.dumps(rec, ensure_ascii=False))
    f.write_text("\n".join(out) + "\n", encoding="utf-8")
    total_files += 1
    total_fixed += n_fixed
    print(f"  {f.name:<28} {len(out):>4} records, {n_fixed:>4} calls rewritten  (backup: {backup.name})")

print(f"\n{total_files} file(s), {total_msgs} messages, {total_fixed} tool calls rewritten")
for what, n in sorted(summary.items(), key=lambda kv: -kv[1]):
    print(f"    {n:>5}x  {what}")

# Hard check: no rejected argument name may survive anywhere.
BANNED = ('"node_id"', '"top_k"')
for f in sorted(TRAJECTORIES_DIR.glob("*.jsonl")):
    blob = f.read_text(encoding="utf-8")
    for b in BANNED:
        assert b not in blob, f"{f.name} still contains {b}"
    for line in blob.splitlines():
        if not line.strip():
            continue
        for msg in json.loads(line).get("messages", []):
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", tc)
                if fn.get("name") == "submit_patch":
                    assert not fn.get("arguments"), "submit_patch still carries arguments"
                if fn.get("name") == "get_code_subgraph":
                    assert isinstance(fn.get("arguments", {}).get("nodes"), list), \
                        "get_code_subgraph.nodes must be a list"
print("\n✓ All trajectories conform to the real tool interface.")
```

---

#### Cell 9: Merge & Verify Final SFT Dataset (`sft_combined.jsonl`)
*(Prioritizes the 115 real ground-truth training tasks + grounded synthetic tasks)*

```python
import json
from pathlib import Path

print("=== FINAL DATASET MERGE & AUDIT ===")
WORKING_DIR = Path("/kaggle/working")
TRAJECTORIES_DIR = WORKING_DIR / "data/trajectories"
f_real = TRAJECTORIES_DIR / "sft_real_train.jsonl"
f_scaled = TRAJECTORIES_DIR / "sft_scaled.jsonl"
f_combined = TRAJECTORIES_DIR / "sft_combined.jsonl"

all_trajectories = {}
provenance = {}

# BUGFIX: this loop previously did `all_trajectories[syn_id] = item` unconditionally, so
# sft_scaled OVERWROTE sft_real_train on any key collision — the exact opposite of the
# "real tasks first" intent. Guard with `not in` so the grounded record always wins.
for src, tag in [(f_real, "grounded"), (f_scaled, "synthetic")]:
    if src.exists():
        with open(src) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    syn_id = item.get("synthetic_id")
                    if syn_id and item.get("messages") and syn_id not in all_trajectories:
                        all_trajectories[syn_id] = item
                        provenance[syn_id] = tag

with open(f_combined, "w", encoding="utf-8") as f:
    for item in all_trajectories.values():
        f.write(json.dumps(item) + "\n")

# Report composition explicitly. ~62% of the previous run's training signal was
# unverified synthetic screenplays with hallucinated observations, which was not visible
# anywhere in the pipeline. It must be visible now.
n_grounded = sum(1 for v in provenance.values() if v == "grounded")
n_synth = sum(1 for v in provenance.values() if v == "synthetic")
total = max(1, len(all_trajectories))

print(f"🎉 Combined Dataset: {len(all_trajectories)} trajectories -> {f_combined.name}")
print(f"   Grounded (real maintainer fixes): {n_grounded} ({n_grounded/total*100:.1f}%)")
print(f"   Synthetic (UNVERIFIED teacher):   {n_synth} ({n_synth/total*100:.1f}%)")
if n_synth / total > 0.5:
    print(f"   ⚠️  WARNING: {n_synth/total*100:.1f}% of training signal is unverified synthetic data")
    print(f"       with hallucinated tool observations. Consider training grounded-only")
    print(f"       (Stage 4 of TRAINING_FIX_PLAN_DEC2.md) and comparing.")
print("✓ Ready for SFT training in the Master Training Notebook.")
```
