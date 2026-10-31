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

### ✅ Milestone 4: Stage 4A QLoRA SFT Post-Training (SFT v1 Completed & Evaluated)
- **Gemma 4 31B SFT Adapter Trained on 4x L4 GPUs:**
  - Rank-32 LoRA (`r=32`, `alpha=64`, lr=$2\times 10^{-4}$) trained across all 410 linear projections of Gemma 4 31B (244.8M trainable parameters / 0.78% of model).
  - Trained for 3 full epochs over 112 verified RepoGraphForge trajectories (42 optimizer steps, effective batch size = 8).
  - **Convergence:** Smooth, textbook loss trajectory from **`2.046` down to `0.3935`** (overall average train loss: `0.7195`).
  - **Adapter Export:** Safely exported `adapter_config.json` and `adapter_model.safetensors` (979.56 MB) to `submission/adapters/main_lora/`.
  - **Runtime & Efficiency:** 1,863 seconds (~31.0 mins), preserving over 25 hours of Kaggle GPU quota.

- **Holdout Evaluation & Empirical Finding (The Text-Tool Serialization Trap):**
  - Evaluated SFT v1 in the live `swegemma` sandbox on holdout tasks (`requests_7505`, `requests_7502`).
  - **Massive Localization Breakthrough:** Model diagnosed root causes in under 15 seconds (vs. 60+ seconds of blind `grep` loops in zero-shot base), identifying exact symbols: `_encode_files`, `is_file_like`, `SupportsRead`, and `PreparedRequest.body`.
  - **Observed Failure Mode:** Model output its tool invocations as raw JSON text strings inside conversational messages (`[{'id': 'call_4', ...}]`) rather than triggering native environment function calls.
  - **Fix A Result:** Prompt steering alone failed to overcome fine-tuned next-token probabilities.
  - **Scientific Conclusion for Paper Track:** Establishes the necessity of strict structural token wrapping (`<tool_call>`) in SFT formatting to bridge the gap between *semantic reasoning* and *environment execution*.

---

## 2. Updated Action Plan: SFT v2 (Native Tool Tokens), Full Benchmark, & Stage 4B RFT

### Step 4.1: Retrain SFT v2 with Native `<tool_call>` Tokens (~25 mins)
- Formatter updated in `src/train_qlora.py` and `src/teacher_rollouts.py` to wrap tool calls in native `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` tokens.
- Re-run SFT training on 4x L4 GPUs using the updated collator.

### Step 4.2: Full Holdout Benchmark (Table 2 Row 2)
- Serve SFT v2 via vLLM with `main_lora`.
- Run evaluation across the full 14-task holdout suite (`requests` + `httpx`) to record **Table 2 Row 2 (SFT Pass@1)**.

### Step 4.3: Stage 4B Best-of-N Rejection Fine-Tuning (RFT Showdown)
- Run `src/rft_sampler.py` to sample $N=4$ rollouts per task on 50 held-out synthetic tasks.
- Run `src/rft_pipeline.py` to partition rollouts across:
  1. $V_{\text{free}}$ (unanchored judge) -> `rft_v_free_accepted.jsonl`
  2. $V_{\text{graph}}$ (GraphForge evidence rubric) -> `rft_v_graph_accepted.jsonl`
  3. $V_{\text{exec}} + V_{\text{graph}}$ (hybrid oracle) -> `rft_hybrid_accepted.jsonl`
- Train low-lr alignment pass (lr=$5\times 10^{-5}$) to complete Table 2 Rows 3, 4, 5.
