# RepoGraphForge: Project Progress Log & Action Plan

**Repository:** `repographforge-gemma4`  
**Current Date:** October 27, 2026  
**Primary Goals:** 
1. Kaggle Code Submission: Autonomous Gemma 4 Developer Agent
2. Kaggle Paper Track: Empirical Study on Code-Graph Trajectories & Evidence Rubrics (Table 1 & Table 2)

---

## 1. Where We Are Right Now

### ✅ Milestone 1: Data Exploration, Anti-Leakage Partition & Baseline (COMPLETED)
- **Local Tooling Implemented:**
  - `src/split_dataset.py`: Partitions the 129 public tasks into training pool ($D_{\text{train}}$) and strict Out-of-Distribution validation ($D_{\text{val-ood}}$).
  - `src/data_loader.py`: Ingestion layer mapping NetworkX multigraphs (`graphs/*.json`) and 256-dim embeddings (`embeddings/*.npz`) to tasks by commit SHA.
  - `src/graph_sampler.py`: Subgraph extraction combining 2-hop structural ego-graphs with semantic embedding neighbors.
- **Repository-Level Split Verified:**
  - **Training & Synthesis Pool ($D_{\text{train}}$):** 115 tasks (`fastapi/fastapi`: 67, `Textualize/rich`: 48).
  - **Out-of-Distribution Holdout ($D_{\text{val-ood}}$):** 14 tasks across two unseen HTTP libraries (`psf/requests`: 13, `encode/httpx`: 1).
- **Kaggle Environment & First Baseline Recorded:**
  - Ryan Holbrook's official starter notebook cloned and verified.
  - Offline dependencies (`swegemma`, `adk_submission`, `adk_eval_core`) installed from the wheelhouse dataset.
  - Gemma 4 12B served on 4x L4 GPUs via vLLM (`tp=4`).
  - **Table 2 Row 1 (Zero-Shot Baseline):** Evaluated on OOD holdout (`requests` smoke test):
    - **Pass@1:** `0.0%` (`NO_PATCH` / agent timeout).
    - **Failure Mode:** Brute-force `grep -r` and `find` trial-and-error in sandbox without localized patch submission.
  - **GPU Quota Preserved:** ~26h 03m of 30h remaining.

---

### ✅ Milestone 2: Evidence Rubrics & Verifier Alignment (COMPLETED!)
- **100% Offline with Gemma 4 31B:**
  - Fully synthesized GraphForge structural contract rubrics ($R_G$) for all 129 tasks using local vLLM on 4x L4 GPUs.
  - Assembled balanced evaluation suite of 386 candidate patches (129 gold passing $y^+$, 257 contract-omission & sycophantic drift failures $y^-$).
- **Table 1 Empirical Findings Confirmed:**
  - Evaluated 386 candidates under both $V_{\text{free}}$ (unanchored judge) and $V_{\text{graph}}$ (GraphForge rubric judge) against ground-truth $V_{\text{exec}}$.
  - **Results:**
    - Recall vs $V_{\text{exec}}$: **0.6512 $\rightarrow$ 0.7442** (+9.3 percentage points / +14.3% relative).
    - F1 Score: **0.7434 $\rightarrow$ 0.7967** (+5.33 points).
    - Cohen's $\kappa$: **0.6401 $\rightarrow$ 0.7051** (crosses into strong substantial agreement).
  - **Scientific Conclusion:** Anchoring LLM judges to code-graph dependency contracts prevents false rejection of working fixes and dramatically aligns offline LLM evaluation with deterministic test execution.

---

### ✅ Milestone 3: Subgraph Sampling & Synthetic Task Generation (COMPLETED!)
- **1,139 Connected Subgraphs Sampled:**
  - `src/graph_sampler.py` extracted 1,139 multi-node subgraphs across `fastapi` and `rich` combining 2-hop structural ego-graphs with 256-dim embedding cosine neighbors.
- **350 Multi-File Synthetic Tasks Generated:**
  - Gemma 4 31B applied reverse fault injection to synthesize paired `problem_statement`, `bug_patch`, `reproduction_test`, and `evidence_rubric` saved to `synthetic_tasks.jsonl`.
- **113 Pristine Teacher Trajectories Collected:**
  - High-density quality filter admitted 113 complete 4-turn graph-first expert trajectories (`sft_verified.jsonl`) ready for SFT QLoRA fine-tuning.

---

## 2. Action Plan for Milestone 4: Gemma 4 QLoRA SFT & RFT Training

Milestone 4 is the training phase where we adapt Gemma 4 with parameter-efficient LoRA adapters:

### Step 4.1: QLoRA Supervised Fine-Tuning (SFT) (~45–60 mins)
- Train a rank-32 LoRA adapter (`r=32`, `alpha=64`, lr=$2\times 10^{-4}$) on `sft_verified.jsonl` using HuggingFace TRL / SFTTrainer on 4x L4 GPUs.
- Export trained adapter weights to `submission/adapters/main_lora/`.

### Step 4.2: Best-of-N Rollout & Rejection Fine-Tuning (RFT)
- Sample candidate rollouts from the SFT checkpoint.
- Filter candidate rollouts using our verified $V_{\text{graph}}$ verifier.
- Low learning-rate RFT alignment pass (lr=$5\times 10^{-5}$).
