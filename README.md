# RepoGraphForge: Post-Training Lightweight Coding Agents via Code-Graph Trajectory Synthesis & Rubric-Guided Rejection Fine-Tuning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Competition: Gemma 4 Developer Agent](https://img.shields.io/badge/Kaggle-Gemma_4_Developer_Agent-orange.svg)](https://www.kaggle.com/competitions/gemma-4-developer-agent)

RepoGraphForge is a research and engineering blueprint adapting **GraphForge** (`arXiv:2609.38923`) to repository-scale software engineering. Developed for the **Google DeepMind - The Gemma 4 Developer Agent Competition & Paper Track**, it leverages pre-computed NetworkX code graphs, 256-dimensional node embeddings, and deterministic test execution oracles to synthesize multi-file bug-fixing tasks, evidence-anchored verification rubrics, and verified teacher trajectories for offline developer agents running on consumer hardware.

---

## Key Highlights & Empirical Findings

1. **Table 1: Verifier Alignment & Sycophantic Patch Drift Analysis**
   - Evaluated 386 candidate patches (129 gold solutions, 257 subtle contract omissions & sycophantic drift diffs) using **Gemma 4 31B** served locally on 4x NVIDIA L4 GPUs.
   - **Recall vs. $V_{\text{exec}}$:** Surged from **0.6512** (Unanchored Judge $V_{\text{free}}$) to **0.7442** (GraphForge Evidence Rubric $V_{\text{graph}}$) — a **+14.28% relative gain**.
   - **F1 Score:** Increased from **0.7434** to **0.7967** (+5.33 points).
   - **Cohen's $\kappa$:** Jumped from **0.6401** to **0.7051**, crossing into strong substantial agreement with the ground-truth deterministic execution oracle ($V_{\text{exec}}$).
2. **100% Offline Reproduction:**
   - The entire synthesis, verification, and post-training pipeline operates locally using open weights and pre-computed graph representations, with zero reliance on external cloud APIs.
3. **Data Efficiency (The LIMA Principle for Agents):**
   - Rather than brute-force memorization of public tasks, RepoGraphForge synthesizes hundreds of multi-node bug-fix tasks via reverse fault injection across connected ego-graphs, curating high-density graph-first navigation trajectories.

---

## Repository Structure

```text
repographforge-gemma4/
├── data/
│   ├── raw/                  # Official dataset (tasks.jsonl, graphs/, embeddings/, wheels/)
│   ├── splits/               # Leak-free train (115) and OOD validation (14) splits + candidate suites
│   ├── synthetic_tasks/      # Sampled subgraphs (1,139) and reverse-mutated tasks (350)
│   └── trajectories/         # Verified 4-turn teacher SFT rollouts (sft_verified.jsonl)
├── src/
│   ├── split_dataset.py      # Milestone 1: Anti-leakage repository partitioner
│   ├── data_loader.py        # Milestone 1: Graph, embedding, and task ingestion manager
│   ├── rubric_verifier.py    # Milestone 2: Mathematical verification metrics & confusion matrix
│   ├── rubric_generator.py   # Milestone 2: Evidence rubric (R_G) synthesizer (vLLM / Gemini)
│   ├── candidate_patch_generator.py # Milestone 2: Balanced candidate evaluation testbed (386 diffs)
│   ├── eval_verifier_alignment.py   # Milestone 2: Table 1 evaluator with multi-threaded vLLM batching
│   ├── graph_sampler.py      # Milestone 3.1: 2-hop structural ego-graph + cosine embedding sampler
│   ├── task_synthesizer.py   # Milestone 3.2A: Reverse fault injection task generator
│   ├── teacher_rollouts.py   # Milestone 3.2B: Graph-first teacher agent rollout collector
│   └── train_qlora.py        # Milestone 4: QLoRA SFT and RFT adapter trainer
├── experiments/
│   ├── logs/                 # Table 1 empirical evaluation logs (JSON)
│   └── plots/                # Data scaling curves & verifier confusion figures
├── submission/               # Evaluation package for Kaggle submission.zip
│   ├── agent.yaml            # Graph-first agent configuration and system prompts
│   └── adapters/             # Trained PEFT LoRA adapters (main_lora/)
├── competition_plan.md       # Master execution blueprint and research roadmap
├── PROGRESS_LOG.md           # Milestone tracking and empirical progress log
└── agent_skill_eval_hackathon.md # 90-minute multi-agent skill regression challenge
```

---

## Quickstart & Reproduction Guide

### 1. Environment Installation
```bash
git clone https://github.com/elena-kalinina/Gemma4_Competition.git repographforge-gemma4
cd repographforge-gemma4

# Install offline wheelhouse packages (on Kaggle or local Linux)
pip install swegemma adk-submission adk-eval-core networkx numpy torch transformers peft trl
```

### 2. Milestone 1: Anti-Leakage Dataset Partitioning
Partitions the 129 official tasks into $D_{\text{train}}$ (115 tasks: `fastapi`, `rich`) and strict Out-of-Distribution validation $D_{\text{val-ood}}$ (14 tasks: `requests`, `httpx`):
```bash
python3 src/split_dataset.py \
    --tasks-path data/raw/tasks.jsonl \
    --output-dir data/splits
```

### 3. Milestone 2: Table 1 Verifier Alignment Study
Synthesize structural contract rubrics ($R_G$), build the 386-diff candidate suite, and evaluate $V_{\text{free}}$ vs. $V_{\text{graph}}$:
```bash
# Generate 129 evidence rubrics
python3 src/rubric_generator.py --backend vllm --vllm-url http://127.0.0.1:8000/v1

# Assemble balanced candidate patch suite (129 gold, 257 drift/omission)
python3 src/candidate_patch_generator.py

# Run Table 1 empirical showdown
python3 src/eval_verifier_alignment.py --backend vllm --workers 4
```

### 4. Milestone 3: Subgraph Sampling & Reverse Task Synthesis
Extract multi-node subgraphs from the pre-computed repository multigraphs and synthesize graph-anchored bug tasks:
```bash
# 3.1 Sample 1,000+ connected subgraphs combining 2-hop topology and node embeddings
python3 src/graph_sampler.py \
    --graphs-dir data/raw/graphs \
    --embeddings-dir data/raw/embeddings \
    --output-file data/synthetic_tasks/sampled_subgraphs.jsonl

# 3.2A Reverse Fault Injection (Gemma 4 31B)
python3 src/task_synthesizer.py \
    --subgraphs-file data/synthetic_tasks/sampled_subgraphs.jsonl \
    --output-file data/synthetic_tasks/synthetic_tasks.jsonl \
    --workers 4

# 3.2B Collect verified graph-first teacher trajectories
python3 src/teacher_rollouts.py \
    --synthetic-tasks data/synthetic_tasks/synthetic_tasks.jsonl \
    --output-file data/trajectories/sft_verified.jsonl \
    --workers 4
```

### 5. Milestone 4: QLoRA SFT Post-Training
Train a parameter-efficient rank-32 adapter on the verified trajectories:
```bash
python3 src/train_qlora.py \
    --model-id google/gemma-4-31b-it-qat-w4a16-ct \
    --dataset-path data/trajectories/sft_verified.jsonl \
    --output-dir submission/adapters/main_lora \
    --lr 2e-4 \
    --epochs 3
```

---

## Citation & Acknowledgments

This research builds upon Google DeepMind's Gemma 4 model family, the Kaggle Gemma 4 Developer Agent benchmark harness (`swegemma`), and GraphForge (`arXiv:2609.38923`).
