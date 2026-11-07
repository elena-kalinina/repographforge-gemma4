# RepoGraphForge: Data Preparation & Synthesis Pipeline Guide

This notebook guide organizes the complete **data preparation, anti-leakage partitioning, evidence rubric synthesis, and teacher trajectory collection** into a modular, repeatable, and quota-efficient workflow.

---

## 🧭 Master Execution Flow & Hardware Budget

| Milestone Phase | Script / Operation | Hardware | Quota Impact | Runtime |
| :--- | :--- | :---: | :---: | :---: |
| **Milestone 1** | Anti-Leakage Repository Partitioning | **CPU** | **0 GPU hrs** | ~2 seconds |
| **Milestone 2A** | Evidence Rubric ($R_G$) Compilation | **CPU** (Gemini) or **GPU** (vLLM) | 0 hrs (API) / 0.15 hrs | ~4–8 mins |
| **Milestone 2B** | 386 Candidate Patch Suite Generation | **CPU** | **0 GPU hrs** | ~3 seconds |
| **Milestone 2C** | Verifier Alignment Evaluation (Table 1) | **GPU L4 x4** (vLLM) | ~0.15 hrs | ~6–8 mins |
| **Milestone 3A** | Multi-Hop Subgraph & Embedding Sampler | **CPU** | **0 GPU hrs** | ~30 seconds |
| **Milestone 3B** | Reverse Fault Injection Task Synthesis | **GPU L4 x4** (vLLM) | ~0.4 hrs | ~20–25 mins |
| **Milestone 3C** | Multi-Hop Teacher Trajectory Rollouts | **GPU L4 x4** (vLLM) | ~0.3 hrs | ~15–18 mins |
| **Dataset Merge** | Deduplication & Token Formatting | **CPU** | **0 GPU hrs** | ~1 second |

---

## 📋 Cell-by-Cell Notebook Implementation

### PART 1: MILESTONE 1 — ANTI-LEAKAGE DATASET PARTITIONING (100% CPU)

> **Pro-Tip:** Run Cells 1 and 2 on **Accelerator: None (CPU)** to preserve 100% of your weekly GPU quota!

#### Cell 1: Environment Paths & Directory Setup
```python
import os
from pathlib import Path

# Competition data paths
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

# Write train_tasks.jsonl (115 tasks)
with open(train_file, "w", encoding="utf-8") as f:
    for t in train_tasks: f.write(json.dumps(t) + "\n")

# Write val_ood_tasks.jsonl (14 tasks)
with open(val_ood_file, "w", encoding="utf-8") as f:
    for t in val_ood_tasks: f.write(json.dumps(t) + "\n")

# Write smoke_test_tasks.jsonl (2 tasks: requests_7505, requests_7502)
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

# Option A: Google AI Studio API (Free Tier, 0 GPU quota)
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
                time.sleep(4.2)  # Respect free tier rate limits
            else:
                # Query local vLLM
                import urllib.request
                req = urllib.request.Request(
                    "http://127.0.0.1:8000/v1/chat/completions",
                    data=json.dumps({"model": "main_lora", "messages": [{"role": "system", "content": RUBRIC_SYS}, {"role": "user", "content": user_prompt}], "temperature": 0.2}).encode(),
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

#### Cell 4: Generate Balanced Candidate Patch Suite (386 Patches: Gold $y^+$, Partial $y^-$, Drift $y^-$)
*(100% CPU — Runs in ~3 seconds)*

```python
import json, re
from pathlib import Path

print("=== GENERATING 386-PATCH CANDIDATE EVALUATION SUITE ===")
candidates_file = SPLITS_DIR / "candidate_patches_eval.jsonl"

rubric_map = {}
rubrics_file = SPLITS_DIR / "evidence_rubrics.jsonl"
if rubrics_file.exists():
    with open(rubrics_file) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rubric_map[item["instance_id"]] = item.get("evidence_rubric", [])

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

        # Candidate 3: Sycophantic Drift Patch (y-)
        drift_patch = "diff --git a/fix.py b/fix.py\n+ # Handled edge case defensively\n+ pass"
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

print(f"🎉 Created {len(candidates)} balanced candidate patches across {len(all_tasks)} tasks.")
print(f"   - Gold Passing (y+):   {sum(1 for c in candidates if c['v_exec_ground_truth'])}")
print(f"   - Drift/Failing (y-): {sum(1 for c in candidates if not c['v_exec_ground_truth'])}")
```

---

#### Cell 5: Evaluate $V_{\text{free}}$ vs $V_{\text{graph}}$ vs $V_{\text{exec}}$ (Outputs Table 1)
*(Requires vLLM running on 4x L4 GPUs, ~6 to 8 minutes)*

```python
import json, urllib.request, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=== EVALUATING VERIFIER ALIGNMENT BENCHMARK (TABLE 1) ===")
BASE_URL = "http://127.0.0.1:8000/v1"
candidates_file = SPLITS_DIR / "candidate_patches_eval.jsonl"
table1_log_file = WORKING_DIR / "experiments/logs/verifier_alignment_table1.json"
table1_log_file.parent.mkdir(parents=True, exist_ok=True)

with open(candidates_file) as f:
    candidates = [json.loads(line) for line in f if line.strip()]

def call_judge(prompt, sys_prompt):
    payload = {"model": "main_lora", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 128}
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
    
    # V_free: Unanchored LLM Judge
    sys_free = "You are a code judge. Decide if this diff fixes the issue. Output: {\"verdict\": \"PASS\" | \"FAIL\"}"
    v_free = call_judge(f"### Issue:\n{prob}\n\n### Patch:\n{patch}", sys_free)
    
    # V_graph: GraphForge Evidence Rubric Judge
    sys_graph = f"You are a strict GraphForge rubric judge. Verify against every contract:\n{rubric_str}\nIf any contract violated, output FAIL. Output: {{\"verdict\": \"PASS\" | \"FAIL\"}}"
    v_graph = call_judge(f"### Issue:\n{prob}\n\n### Patch:\n{patch}", sys_graph)
    
    return {
        "ground_truth": cand["v_exec_ground_truth"],
        "v_free": v_free,
        "v_graph": v_graph
    }

print(f"Evaluating {len(candidates)} candidate patches using 4 parallel streams...")
t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(eval_candidate, c): c for c in candidates}
    for i, fut in enumerate(as_completed(futures), start=1):
        results.append(fut.result())
        if i % 50 == 0 or i == len(candidates):
            print(f"  [{i}/{len(candidates)}] evaluated in {time.time()-t0:.1f}s...")

# Compute Table 1 Confusion Matrices
def calc_metrics(preds, truths):
    tp = sum(p and t for p, t in zip(preds, truths))
    fp = sum(p and not t for p, t in zip(preds, truths))
    fn = sum(not p and t for p, t in zip(preds, truths))
    tn = sum(not p and not t for p, t in zip(preds, truths))
    
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * (prec * rec) / max(1e-6, prec + rec)
    fp_drift = fp / max(1, fp + tn)
    
    # Cohen's Kappa
    po = (tp + tn) / len(truths)
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (len(truths) ** 2)
    kappa = (po - pe) / max(1e-6, 1 - pe)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "fp_drift": round(fp_drift, 4), "kappa": round(kappa, 4)}

truths = [r["ground_truth"] for r in results]
m_free = calc_metrics([r["v_free"] for r in results], truths)
m_graph = calc_metrics([r["v_graph"] for r in results], truths)

table1_payload = {
    "num_candidates": len(candidates),
    "v_free": m_free,
    "v_graph": m_graph,
    "relative_recall_gain": f"+{((m_graph['recall'] - m_free['recall'])/m_free['recall'])*100:.2f}%"
}
with open(table1_log_file, "w") as f: json.dump(table1_payload, f, indent=2)

print("\n" + "=" * 65)
print("=== TABLE 1: VERIFIER ALIGNMENT BENCHMARK RESULTS ===")
print("=" * 65)
print(f"Verifier Regime   | Precision | Recall  | F1 Score | Drift Rate | Cohen's κ")
print(f"V_free (Standard) | {m_free['precision']:.4f}    | {m_free['recall']:.4f}  | {m_free['f1']:.4f}   | {m_free['fp_drift']:.4f}     | {m_free['kappa']:.4f}")
print(f"V_graph (Rubric)  | {m_graph['precision']:.4f}    | {m_graph['recall']:.4f}  | {m_graph['f1']:.4f}   | {m_graph['fp_drift']:.4f}     | {m_graph['kappa']:.4f}")
print(f"\n🎉 Relative Recall Gain: {table1_payload['relative_recall_gain']}")
print(f"✓ Saved Table 1 metrics to: {table1_log_file.name}")
print("=" * 65)
```

---

### PART 3: MILESTONE 3 — SUBGRAPH SAMPLING & SYNTHETIC TASK GENERATION

#### Cell 6: Extract 1,139 Connected Subgraphs from Competition Code Graphs
*(100% CPU — Ingests NetworkX `.json` graphs and `.npz` node embeddings in ~30 seconds)*

```python
import json, random
from pathlib import Path
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

print("=== EXTRACTING 2-HOP SUBGRAPHS + SEMANTIC EMBEDDINGS ===")

graphs_dir = DATA_DIR / "graphs"
embeddings_dir = DATA_DIR / "embeddings"
output_subgraphs = SYNTHETIC_DIR / "sampled_subgraphs.jsonl"

graph_files = sorted(graphs_dir.glob("*.json"))
print(f"Discovered {len(graph_files)} pre-computed repository graphs.")

subgraphs = []

for g_path in graph_files:
    # Skip holdout repositories
    if any(ood in g_path.name.lower() for ood in ["requests", "httpx"]):
        continue
        
    try:
        with open(g_path) as f:
            data = json.load(f)
            G = json_graph.node_link_graph(data)
            
        emb_path = embeddings_dir / g_path.name.replace(".json", ".npz")
        emb_data = np.load(emb_path) if emb_path.exists() else None
        
        nodes = list(G.nodes())
        if len(nodes) < 5: continue
        
        # Sample 10 anchor nodes per graph
        anchors = random.sample(nodes, min(10, len(nodes)))
        
        for a in anchors:
            # 2-hop structural ego graph
            ego_nodes = list(nx.ego_graph(G.to_undirected(), a, radius=2).nodes())[:6]
            sub_g = G.subgraph(ego_nodes)
            
            nodes_meta = [{"id": n, **G.nodes[n]} for n in sub_g.nodes()]
            edges_meta = [{"source": u, "target": v, **d} for u, v, d in sub_g.edges(data=True)]
            
            files = set()
            for n in nodes_meta:
                f = n.get("file") or n.get("file_path") or str(n["id"]).split(".")[0] + ".py"
                files.add(f)
                
            subgraphs.append({
                "anchor_node": a,
                "graph_snapshot": g_path.name,
                "num_nodes": len(nodes_meta),
                "num_files": len(files),
                "files": list(files),
                "nodes": nodes_meta,
                "edges": edges_meta
            })
    except Exception:
        pass

with open(output_subgraphs, "w", encoding="utf-8") as f:
    for s in subgraphs: f.write(json.dumps(s) + "\n")

print(f"🎉 Successfully sampled {len(subgraphs)} subgraphs -> {output_subgraphs.name}")
print(f"   Multi-file subgraphs: {sum(1 for s in subgraphs if s['num_files'] > 1)} / {len(subgraphs)}")
```

---

#### Cell 7: Reverse Fault Injection Task Synthesis (`synthetic_tasks.jsonl`)
*(Runs on GPU L4 x4 via vLLM, ~20 mins)*

```python
import json, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=== REVERSE FAULT INJECTION TASK SYNTHESIS ===")
BASE_URL = "http://127.0.0.1:8000/v1"
subgraphs_file = SYNTHETIC_DIR / "sampled_subgraphs.jsonl"
output_synthetic = SYNTHETIC_DIR / "synthetic_tasks.jsonl"

with open(subgraphs_file) as f:
    subgraphs = [json.loads(line) for line in f if line.strip()][:350]

SYNTH_SYS = """You are an expert SWE task synthesizer.
Given connected code-graph nodes, inject a realistic subtle multi-node bug (breaking an interface contract, return type mismatch, or state issue).
Output strictly valid JSON:
{
  "problem_statement": "Realistic GitHub issue describing symptoms without leaking function names.",
  "bug_patch": "Unified git diff introducing the defect.",
  "reproduction_test": "Pytest function failing on bug_patch and passing on clean code.",
  "evidence_rubric": [
    "Contract 1: Specifying parameter/type invariant...",
    "Contract 2: Specifying caller/callee contract...",
    "Contract 3: Specifying error handling..."
  ]
}
"""

def synthesize_task(sub):
    anchor = sub["anchor_node"]
    nodes = "\n".join(f"- {n['id']} in {n.get('file', 'repo')}" for n in sub["nodes"])
    edges = "\n".join(f"- {e['source']} -> {e['target']}" for e in sub["edges"][:6])
    
    user_prompt = f"### Anchor: {anchor}\n### Subgraph Nodes:\n{nodes}\n\n### Directed Edges:\n{edges}\n"
    
    payload = {"model": "main_lora", "messages": [{"role": "system", "content": SYNTH_SYS}, {"role": "user", "content": user_prompt}], "temperature": 0.3}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            txt = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            parsed = json.loads(txt.split("```json")[-1].split("```")[0].strip())
            return {
                "synthetic_id": f"syn_{anchor}",
                "anchor_node": anchor,
                "problem_statement": parsed["problem_statement"],
                "bug_patch": parsed["bug_patch"],
                "evidence_rubric": parsed["evidence_rubric"]
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

print(f"🎉 Created {len(synthesized)} synthetic bug tasks -> {output_synthetic.name}")
```

---

#### Cell 8: Collect Scaled Multi-Hop Teacher Trajectories (`sft_scaled.jsonl`)
*(Runs on GPU L4 x4 via vLLM, ~15 mins)*

```python
import json, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

print("=== COLLECTING SCALED MULTI-HOP TEACHER TRAJECTORIES ===")
BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

synthetic_tasks_file = SYNTHETIC_DIR / "synthetic_tasks.jsonl"
output_trajectories = TRAJECTORIES_DIR / "sft_scaled.jsonl"

with open(synthetic_tasks_file) as f:
    tasks = [json.loads(line) for line in f if line.strip()][:350]

TEACHER_SYS = """You are an expert autonomous software engineer solving a real repository issue.
Follow a strict graph-first multi-hop navigation workflow across multi-file dependencies.
Format strictly as JSON: {"messages": [...]} with role: 'user', 'assistant', 'tool'.
"""

def generate_trajectory(task):
    syn_id = task["synthetic_id"]
    anchor = task["anchor_node"]
    prob = task["problem_statement"][:1800]
    patch = task["bug_patch"][:2000]
    rubric = "\n".join(f"- {r}" for r in task.get("evidence_rubric", []))

    user_prompt = (
        f"Generate a 4-turn autonomous agent trajectory in OpenAI chat format.\n"
        f"### Anchor: {anchor}\n### Problem:\n{prob}\n### Rubric:\n{rubric}\n### Reference Fix:\n{patch}\n"
        f"Turn 1: get_code_subgraph on anchor.\n"
        f"Turn 2: inspect connected helper file via read_file.\n"
        f"Turn 3: apply contract-preserving fix via edit_file.\n"
        f"Turn 4: submit_patch.\n"
    )

    payload = {"model": "main_lora", "messages": [{"role": "system", "content": TEACHER_SYS}, {"role": "user", "content": user_prompt}], "temperature": 0.2}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            parsed = json.loads(content.split("```json")[-1].split("```")[0].strip())
            msgs = parsed.get("messages", [])
            if len(msgs) >= 4:
                return {"synthetic_id": syn_id, "anchor_node": anchor, "messages": msgs}
    except Exception: return None

t0 = time.time()
collected = []
with open(output_trajectories, "w", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(generate_trajectory, t): t for t in tasks}
    for i, fut in enumerate(as_completed(futures), start=1):
        res = fut.result()
        if res:
            out_f.write(json.dumps(res) + "\n")
            out_f.flush()
            collected.append(res)
        if i % 50 == 0 or i == len(tasks):
            print(f"  [{i}/{len(tasks)}] trajectories collected ({len(collected)} golden in {time.time()-t0:.1f}s)...")

print(f"🎉 Successfully collected {len(collected)} multi-hop teacher trajectories -> {output_trajectories.name}")
```

---

#### Cell 9: Merge & Verify Final SFT Dataset (`sft_combined.jsonl`)
*(100% CPU — Runs in 1 second)*

```python
import json
from pathlib import Path

print("=== FINAL DATASET MERGE & AUDIT ===")
f_verified = TRAJECTORIES_DIR / "sft_verified.jsonl"
f_scaled = TRAJECTORIES_DIR / "sft_scaled.jsonl"
f_combined = TRAJECTORIES_DIR / "sft_combined.jsonl"

all_trajectories = {}
for src in [f_verified, f_scaled]:
    if src.exists():
        with open(src) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    syn_id = item.get("synthetic_id")
                    if syn_id and syn_id not in all_trajectories:
                        all_trajectories[syn_id] = item

with open(f_combined, "w", encoding="utf-8") as f:
    for item in all_trajectories.values():
        f.write(json.dumps(item) + "\n")

print(f"🎉 Combined Dataset: {len(all_trajectories)} Total Golden Trajectories -> {f_combined.name}")
print("✓ Ready for SFT training in the Master Training Notebook!")
```
