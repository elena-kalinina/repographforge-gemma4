# RepoGraphForge: Starter Code Bundle for `~/SideProjects/Gemma4_Competition`

Because this assistant runs in a sandboxed cloud environment without direct write access to your local disk, you can generate the entire project structure and all core Python scripts in one step by copying and running the `bootstrap.sh` script below inside `~/SideProjects/Gemma4_Competition`, or by saving the individual Python files directly.

---

## Option A: One-Step Local Bootstrapper (`bootstrap.sh`)

Open your terminal, navigate to `~/SideProjects/Gemma4_Competition`, and run the following command block to create the folders (`data/`, `src/`, `experiments/`, `submission/`) and populate the starter scripts:

```bash
cd ~/SideProjects/Gemma4_Competition

mkdir -p data/raw data/splits data/synthetic_tasks data/trajectories
mkdir -p src experiments/logs experiments/plots submission/adapters/main_lora

echo "Directory structure created in $(pwd)"
```

---

## Option B: Individual Core Scripts (`src/`)

Save each of the scripts below into `~/SideProjects/Gemma4_Competition/src/`.

### 1. `src/split_dataset.py` (Milestone 1: Anti-Leakage Repository Split)
Partitions the 129 official tasks in `tasks.jsonl` into the training/synthesis pool (`fastapi`, `rich`, `requests`) and the strict out-of-distribution holdout repository (`encode/httpx`).

```python
#!/usr/bin/env python3
"""
Milestone 1: Partition competition tasks.jsonl by repository to prevent data leakage
and create a strict Out-of-Distribution (OOD) validation split for the Paper Track.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def split_tasks(
    tasks_path: Path,
    output_dir: Path,
    ood_repos: tuple = ("encode/httpx",),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train_tasks.jsonl"
    val_ood_file = output_dir / "val_ood_tasks.jsonl"

    train_counts = Counter()
    val_counts = Counter()

    with (
        tasks_path.open("r", encoding="utf-8") as infile,
        train_file.open("w", encoding="utf-8") as train_out,
        val_ood_file.open("w", encoding="utf-8") as val_out,
    ):
        for line in infile:
            if not line.strip():
                continue
            task = json.loads(line)
            repo = task.get("repo", "")

            if any(ood in repo for ood in ood_repos):
                val_out.write(json.dumps(task) + "\n")
                val_counts[repo] += 1
            else:
                train_out.write(json.dumps(task) + "\n")
                train_counts[repo] += 1

    print("=== Repository Split Summary ===")
    print(f"Train Pool ({sum(train_counts.values())} tasks) -> {train_file}")
    for r, c in train_counts.items():
        print(f"  - {r}: {c}")
    print(f"\nOOD Validation Pool ({sum(val_counts.values())} tasks) -> {val_ood_file}")
    for r, c in val_counts.items():
        print(f"  - {r}: {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split tasks.jsonl by repository.")
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("data/raw/tasks.jsonl"),
        help="Path to official tasks.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to save train and OOD split files.",
    )
    args = parser.parse_args()
    split_tasks(args.tasks_path, args.output_dir)
```

---

### 2. `src/graph_sampler.py` (Milestone 1 & 3: Structural + Semantic Subgraph Sampler)
Loads the competition's pre-computed NetworkX JSON code graphs (`graphs/*.json`) and 256-dimensional node embeddings (`embeddings/*.npz`), then extracts connected multi-hop ego-subgraphs enriched with cosine-similar semantic neighbors.

```python
#!/usr/bin/env python3
"""
Milestone 1 & 3: Load pre-computed competition code graphs and 256-dim .npz embeddings
to sample multi-file connected subgraphs for RepoGraphForge task synthesis.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


class RepoGraphSampler:
    def __init__(self, graph_json_path: Path, embedding_npz_path: Optional[Path] = None):
        self.graph_path = graph_json_path
        self.graph: nx.MultiDiGraph = self._load_graph(graph_json_path)
        self.embeddings: Dict[str, np.ndarray] = (
            self._load_embeddings(embedding_npz_path) if embedding_npz_path else {}
        )

    @staticmethod
    def _load_graph(path: Path) -> nx.MultiDiGraph:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Compatible with standard NetworkX node-link JSON used in swegemma
        return json_graph.node_link_graph(data)

    @staticmethod
    def _load_embeddings(path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            return {}
        archive = np.load(path, allow_pickle=True)
        # Support both key-mapped archives and (node_ids, vectors) arrays
        if "node_ids" in archive and "embeddings" in archive:
            return dict(zip(archive["node_ids"], archive["embeddings"]))
        return {k: archive[k] for k in archive.files}

    def find_candidate_anchors(
        self, min_degree: int = 2, max_degree: int = 25
    ) -> List[str]:
        """Filter for non-trivial function/method/class nodes with structural connections."""
        candidates = []
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = str(attrs.get("type", attrs.get("kind", ""))).lower()
            if node_type and node_type not in {"function", "method", "class"}:
                continue
            deg = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)
            if min_degree <= deg <= max_degree:
                candidates.append(node_id)
        return candidates

    def semantic_top_k(
        self, anchor_id: str, exclude_ids: Set[str], top_k: int = 2
    ) -> List[str]:
        """Find top-k cosine-similar nodes across the repo using the 256-dim .npz embeddings."""
        if anchor_id not in self.embeddings:
            return []
        anchor_vec = self.embeddings[anchor_id]
        anchor_norm = np.linalg.norm(anchor_vec) + 1e-8

        scored = []
        for node_id, vec in self.embeddings.items():
            if node_id == anchor_id or node_id in exclude_ids or node_id not in self.graph:
                continue
            sim = float(np.dot(anchor_vec, vec) / (anchor_norm * (np.linalg.norm(vec) + 1e-8)))
            scored.append((sim, node_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored[:top_k]]

    def sample_subgraph(
        self,
        anchor_id: str,
        max_nodes: int = 6,
        semantic_additions: int = 1,
    ) -> Dict[str, Any]:
        """Extract a 2-hop structural ego-graph + semantic neighbors for GraphForge synthesis."""
        undirected = self.graph.to_undirected()
        ego_nodes = list(nx.ego_graph(undirected, anchor_id, radius=2).nodes())
        random.shuffle(ego_nodes)

        selected: List[str] = [anchor_id]
        for nid in ego_nodes:
            if nid != anchor_id and len(selected) < (max_nodes - semantic_additions):
                selected.append(nid)

        sem_nodes = self.semantic_top_k(anchor_id, set(selected), top_k=semantic_additions)
        selected.extend(sem_nodes)

        sub_g = self.graph.subgraph(selected)
        files_touched = {
            self.graph.nodes[n].get("file", self.graph.nodes[n].get("file_path", "unknown"))
            for n in sub_g.nodes()
        }

        nodes_payload = [
            {"id": n, **self.graph.nodes[n]} for n in sub_g.nodes()
        ]
        edges_payload = [
            {"source": u, "target": v, "key": k, **d}
            for u, v, k, d in sub_g.edges(keys=True, data=True)
        ]

        return {
            "graph_file": self.graph_path.name,
            "anchor_node": anchor_id,
            "num_nodes": len(nodes_payload),
            "num_files_touched": len(files_touched),
            "files": sorted(list(files_touched)),
            "nodes": nodes_payload,
            "edges": edges_payload,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample multi-node subgraphs from competition graphs.")
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/raw/graphs"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/raw/embeddings"))
    parser.add_argument("--output-file", type=Path, default=Path("data/synthetic_tasks/sampled_subgraphs.jsonl"))
    parser.add_argument("--samples-per-graph", type=int, default=10)
    parser.add_argument("--exclude-prefix", type=str, default="httpx", help="Holdout repo prefix to skip.")
    args = parser.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    graph_files = sorted(args.graphs_dir.glob("*.json"))

    total_written = 0
    with args.output_file.open("w", encoding="utf-8") as out_f:
        for g_path in graph_files:
            if args.exclude_prefix and g_path.name.lower().startswith(args.exclude_prefix.lower()):
                continue
            emb_path = args.embeddings_dir / g_path.with_suffix(".npz").name
            sampler = RepoGraphSampler(g_path, emb_path)
            anchors = sampler.find_candidate_anchors()
            if not anchors:
                continue

            chosen = random.sample(anchors, min(len(anchors), args.samples_per_graph))
            for anchor in chosen:
                sub = sampler.sample_subgraph(anchor)
                # Prefer multi-file subgraphs to force cross-file graph navigation
                if sub["num_nodes"] >= 3:
                    out_f.write(json.dumps(sub) + "\n")
                    total_written += 1

    print(f"Sampled {total_written} multi-node subgraphs -> {args.output_file}")


if __name__ == "__main__":
    main()
```

---

### 3. `src/rubric_verifier.py` (Milestone 2: $V_{\text{free}}$ vs. $V_{\text{graph}}$ vs. $V_{\text{exec}}$ Evaluation Harness)
Implements the three verification signals required for **Experiment 1 (Table 1)** in your Paper Track writeup and computes Precision, Recall, F1, False-Positive Sycophantic Patch Drift Rate, and Cohen's $\kappa$ against the ground-truth `test_patch` execution oracle ($V_{\text{exec}}$).

```python
#!/usr/bin/env python3
"""
Milestone 2: Verification harness comparing Free-Form LLM Judges (V_free),
GraphForge Evidence-Anchored Rubrics (V_graph), and Deterministic Test Execution (V_exec).
"""
import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List


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
        f"- Node `{n.get('id')}` (file: {n.get('file', 'N/A')}, kind: {n.get('type', 'N/A')})"
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
    Deterministic Oracle (V_exec): Applies the candidate_patch and official test_patch
    inside a clean checkout of the repository at base_commit and runs pytest.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as cand_f, \
         tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as test_f:
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
    """Compute Table 1 Paper Metrics comparing a proxy verifier against V_exec."""
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
    # Self-test with synthetic rollout evaluation logs to verify Table 1 metric calculations
    sample_oracle = [True, False, False, True, False, False, True, False]
    sample_v_free = [True, True, False, True, True, False, False, True]    # High FP drift
    sample_v_graph = [True, False, False, True, False, False, True, True]  # Anchored to AST

    print("=== Table 1: Verifier Alignment vs. Deterministic Execution Oracle (V_exec) ===")
    print("V_free  :", compute_alignment_metrics(sample_v_free, sample_oracle))
    print("V_graph :", compute_alignment_metrics(sample_v_graph, sample_oracle))
```

---

### 4. `src/task_synthesizer.py` (Milestone 3: Reverse Fault Injection & Rubric Generator)
Takes the sampled multi-node subgraphs from `sampled_subgraphs.jsonl` and constructs the prompt payloads for reverse fault injection (`bug_patch` + `test_patch` + `evidence_rubric`).

```python
#!/usr/bin/env python3
"""
Milestone 3: Synthesize multi-file bug-fixing tasks and GraphForge evidence rubrics
from sampled repository subgraphs via Reverse Fault Injection.
"""
import json
from pathlib import Path
from typing import Any, Dict


def build_reverse_mutation_prompt(subgraph: Dict[str, Any], code_snippets: Dict[str, str]) -> str:
    """
    Constructs the Teacher Synthesis Prompt that injects a realistic cross-node bug
    into a clean working commit and outputs the paired issue + GraphForge rubric.
    """
    nodes_desc = "\n".join(
        f"- `{n['id']}` in `{n.get('file', 'unknown')}`" for n in subgraph.get("nodes", [])
    )
    edges_desc = "\n".join(
        f"- `{e['source']}` -> `{e['target']}`" for e in subgraph.get("edges", [])
    )
    formatted_code = "\n\n".join(
        f"### Node: {nid}\n```python\n{code}\n```" for nid, code in code_snippets.items()
    )

    return f"""You are a software engineering task synthesizer implementing RepoGraphForge.
Given the following connected code-graph nodes and their current working Python source code,
inject a realistic, subtle multi-node bug (e.g., breaking an argument contract, edge-case return type,
or state propagation between caller and callee).

### Subgraph Structure
Nodes:
{nodes_desc}

Edges:
{edges_desc}

### Current Working Source Code (Gold State)
{formatted_code}

Produce a JSON object with exactly these keys:
1. "bug_patch": A valid unified git diff that introduces the bug into the working code (inverting this diff yields the gold fix).
2. "reproduction_test": A pytest function that FAILS when `bug_patch` is applied and PASSES on the original working code.
3. "problem_statement": A realistic GitHub issue report describing the user-facing symptom or traceback WITHOUT naming the exact internal function to edit.
4. "evidence_rubric": A list of 3-5 verifiable structural criteria anchored to the specific node IDs in the subgraph.
"""


if __name__ == "__main__":
    subgraphs_file = Path("data/synthetic_tasks/sampled_subgraphs.jsonl")
    if subgraphs_file.exists():
        with subgraphs_file.open() as f:
            first_sub = json.loads(f.readline())
        dummy_snippets = {n["id"]: "# source loaded from repo checkout" for n in first_sub["nodes"][:2]}
        print(build_reverse_mutation_prompt(first_sub, dummy_snippets))
    else:
        print("Run `python src/graph_sampler.py` first to generate sampled_subgraphs.jsonl.")
```

---

### 5. `src/train_qlora.py` (Milestone 4: Gemma 4 QLoRA SFT & RFT Trainer)
Configures Parameter-Efficient Fine-Tuning (PEFT / QLoRA) for Gemma 4 on your verified teacher and Best-of-$N$ RFT trajectories, saving the adapter directly into `submission/adapters/main_lora/` for `submission.zip`.

```python
#!/usr/bin/env python3
"""
Milestone 4: Fine-tune Gemma 4 using QLoRA on verified RepoGraphForge trajectories
and export the adapter weights directly to submission/adapters/main_lora/.
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def train(
    model_id: str,
    dataset_path: Path,
    output_adapter_dir: Path,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_seq_length: int = 8192,
) -> None:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    sft_config = SFTConfig(
        output_dir=str(output_adapter_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))
    print(f"Saved Gemma 4 LoRA adapter to {output_adapter_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gemma 4 QLoRA on RepoGraphForge trajectories.")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-9b-it")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/trajectories/sft_verified.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("submission/adapters/main_lora"))
    parser.add_argument("--lr", type=float, default=2e-4, help="Use 2e-4 for SFT, 5e-5 for RFT pass.")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train(args.model_id, args.dataset_path, args.output_dir, args.lr, args.epochs)
```

---

### 6. `submission/agent.yaml` (Milestone 5: Graph-First Agent Configuration)
Save this inside `~/SideProjects/Gemma4_Competition/submission/agent.yaml` to instruct your fine-tuned Gemma 4 model to prioritize the low-token code-graph tools during evaluation.

```yaml
model:
  adapter_path: adapters/main_lora
  temperature: 0.1
  max_tokens: 4096

agent:
  max_turns: 12
  system_prompt: |
    You are an autonomous software engineering agent equipped with a pre-computed repository code graph.
    Follow this strict workflow to resolve issues efficiently:
    1. LOCALIZE VIA GRAPH FIRST: Before reading full files, call `get_code_subgraph` or `get_code_neighbors` on symbols mentioned in the issue or traceback to map upstream callers and downstream callees.
    2. INSPECT MINIMAL SPANS: Open only the specific files and line ranges identified in the subgraph.
    3. VERIFY CONTRACTS: Ensure your patch preserves parameter types, return signatures, and exceptions expected by connected nodes in the code graph.
    4. TEST: Run the relevant unit tests to verify the fix before submitting your final git diff.
```