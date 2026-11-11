# Plan 1: 5–6 Hour Paper Track Rescue Plan (Deadline: Nov 12, 2026)

**Goal:** Transform `paper/manuscript.md` from an apologetic 6-bug post-mortem into a rigorous, submittable research paper with defensible empirical claims and clean ablations within **5–6 hours**.

**Core Reframing Thesis:**
> **"The Loss–Execution Paradox in Offline Coding Agents: Graph-Anchored Verification, Localization Compression, and Structural Invariants in Post-Training Gemma 4"**
>
> Instead of framing the paper around *"we made 6 scripting mistakes in our pipeline,"* frame it around **three genuine ML/systems findings**:
> 1. **Graph-Guided Localization & Context Compression (Positive Result 1):** Teaching Gemma 4 31B to query 2-hop repository multigraphs (`get_code_subgraph`) before reading files increases unseen-repository bug localization from **21.4% to 85.7%–92.9%** while cutting context consumption from **114.9k tokens (sequential file paging) to 54.4k tokens (>52% reduction)** (the *File-Reading Quicksand Law*).
> 2. **Graph-Anchored Evidence Rubrics vs. Unanchored Judges (Positive Result 2 — Table 1):** Unanchored LLM judges ($V_{\text{free}}$) suffer from high false rejection on multi-hunk fixes and high false acceptance on sycophantic/partial edits that break downstream callers. Anchoring judges to 2-hop AST caller/callee rubrics ($V_{\text{graph}}$) boosts recall by **+14.3% relative** and raises agreement with `pytest` ($V_{\text{exec}}$) to **$\kappa = 0.7051$**.
> 3. **The Loss–Execution Paradox in Agentic Post-Training (Mechanistic Contribution — 3 Structural Phenomena):** Cross-entropy training loss ($2.05 \to 0.35$) is provably blind to catastrophic policy degradation across three generalizable failure mechanisms:
>    - *(M1) Control-Token Gradient Starvation in Folded Templates:* Masking `<turn|>` changes sequence loss by $<0.5\%$ while causing 100% turn-termination failure (run-on generation to `max_output_tokens`).
>    - *(M2) Synthetic Action-Schema Hallucination vs. Grounded Diffs:* Unverified teacher rollouts invent unexecutable tool schemas in **30.5%** of trajectories (vs. **0.0%** in maintainer-grounded trajectories), producing the lowest training loss (**0.349**) alongside **0.0%** resolution.
>    - *(M3) Low-Yield Rejection Overwriting:* Filtering rollouts on surface text patterns or small survivor pools ($n=4$) overwrites SFT tool-calling priors in 4 optimizer steps while doubling sequence loss ($0.64 \to 1.07$).

---

## Choose Your Execution Path (5–6 Hours Total)

- [ ] **Path A (Recommended — ~2.5h GPU on Kaggle + ~2.5h Writing):** Run the clean 114-task Grounded-Only SFT + 14-task holdout evaluation and the clean Table 1 re-run, then update `paper/manuscript.md`.
- [ ] **Path B (Zero-GPU Fallback — ~30m CPU on Kaggle + ~3.5h Writing):** Run only CPU trace forensics (Cells 12A/12B) on existing holdout traces and restructure `paper/manuscript.md` using already-collected artifacts.

---

## Hour 1: Fast CPU Trace Forensics & Dataset Prep on Kaggle (~30–45 mins)

*(Applies to BOTH Path A and Path B)*

- [ ] **1.1 Run Cell 12A (Eval-Trace Schema Discovery)** in `notebooks/repographforge_clean_master.md` (lines 2448–2483) on Kaggle to inspect the JSON structure of `traces/trace_<task>.json`.
- [ ] **1.2 Run Cell 12B (Per-Task Trace Breakdown)** in `notebooks/repographforge_clean_master.md` (lines 2495–2577).
  - [ ] Record `opened with a graph tool: X/14`
  - [ ] Record `called edit_file at all: Y/14`
  - [ ] Record `called submit_patch: Z/14`
  - [ ] Record `submitted with NO edit: W/14` (direct empirical readout of interface drift)
  - [ ] Save the printed table output to `experiments/logs/holdout_trace_forensics.txt`.
- [ ] **1.3 Verify `sft_real_train.jsonl` (114 Grounded Trajectories)** is present and F6-clean in `/kaggle/working/data/trajectories/sft_real_train.jsonl` (already confirmed at 0% discard rate in Cell 8C v3).

---

## Hours 2–3 (Path A Only): Clean Table 1 Re-Run & Grounded-Only SFT Ablation (~1.5–2h GPU)

### Step 2A: Re-Run Table 1 with Task-Specific Negatives & Base 31B Judge (~40 mins)
Removes all "provisional" and "contaminated negative class" caveats from Section 6 of the paper.

- [ ] **2A.1 Generate Task-Specific Candidate Suite (CPU, ~10 seconds):**
  Run `src/candidate_patch_generator.py` on Kaggle so `sycophantic_drift` patches mutate the actual target file and hunk lines of each task rather than using a constant 3-line string:
  ```bash
  python3 src/candidate_patch_generator.py \
    --tasks-file /kaggle/input/gemma-4-developer-agent/tasks.jsonl \
    --rubrics-file /kaggle/working/data/splits/evidence_rubrics.jsonl \
    --output-file /kaggle/working/data/splits/candidate_eval_suite_v2.jsonl
  ```
- [ ] **2A.2 Evaluate with Uncontaminated Base Gemma 4 31B Judge (~35 mins on vLLM):**
  Ensure vLLM is serving the **base** model `gemma-4-31b-it-qat-w4a16-ct` (not `main_lora`) and run:
  ```bash
  python3 src/eval_verifier_alignment.py \
    --candidates-file /kaggle/working/data/splits/candidate_eval_suite_v2.jsonl \
    --output-log /kaggle/working/experiments/logs/verifier_alignment_table1_v2.json \
    --model-name gemma-4-31b-it-qat-w4a16-ct \
    --workers 4
  ```
- [ ] **2A.3 Compute Stratified False-Positive Rates (~2 mins CPU):**
  Run this snippet to report False-Positive rates separately for `partial_contract_violation` and `sycophantic_drift`:
  ```python
  import json
  data = json.load(open("/kaggle/working/experiments/logs/verifier_alignment_table1_v2.json"))
  recs = data["records"]
  for ctype in ("gold_fix", "partial_contract_violation", "sycophantic_drift"):
      sub = [r for r in recs if r["candidate_type"] == ctype]
      n = max(1, len(sub))
      free_pos = sum(1 for r in sub if r["v_free"]) / n
      graph_pos = sum(1 for r in sub if r["v_graph"]) / n
      print(f"{ctype:<28} (n={len(sub):>3}): V_free PASS={free_pos:.2%}, V_graph PASS={graph_pos:.2%}")
  ```
- [ ] **2A.4 Copy `verifier_alignment_table1_v2.json`** back to `experiments/logs/` in this repo.

### Step 2B: Train Clean Grounded-Only SFT (`sft_real_train.jsonl`, 114 Tasks) + Eval (~45 mins)
Tests the clean pipeline (F1–F6 fixed, 0% hallucinated synthetic trajectories) on the 14 OOD holdout tasks.

- [ ] **2B.1 Refresh the Schema Verification Receipt (CPU, ~1 min):**
  In `notebooks/repographforge_clean_master.md`, run:
  - [ ] **Cell 4** (writes `chat_format.py` version `2026-11-10.stage1e`)
  - [ ] **Cell 4B** (verifies version gate)
  - [ ] **Cell 0F** $\to$ **Cell 0C** (writes clean receipt `.tool_schema_verified.json`)
- [ ] **2B.2 Point Cell 6 Strictly at `sft_real_train.jsonl`:**
  In Cell 6 (`train_sft.py`), set:
  ```python
  TRAJECTORIES_FILE = Path("/kaggle/working/data/trajectories/sft_real_train.jsonl")
  ```
  *(This trains exclusively on the 114 grounded maintainer fixes and excludes the unverified synthetic trajectories).*
- [ ] **2B.3 Run Cell 7 (`!python3 /kaggle/working/train_sft.py`, ~18 mins on 4× L4):**
  - [ ] Verify Phase 1.2 gate output at top of log (`supervised fraction` between 3% and 45%, `<turn|>` supervised, 0 observation leaks).
  - [ ] Record final `train_loss` and runtime.
- [ ] **2B.4 Run Holdout Evaluation on the 14 Unseen Tasks (~25 mins):**
  - [ ] Run **Cell 8** with `EVAL_ARM = "C"` (`max_output_tokens: 4096`, budget prompt).
  - [ ] Start vLLM (**Cell 9**) and run the 14-task evaluation loop (**Cell 10**).
  - [ ] Run **Cell 11** (Corrected Audit Table) and **Cell 12B** (Trace Forensics).
  - [ ] Record Row 8 of Table 2: `Strict Submit Rate`, `Applies Cleanly`, `Pytest Resolved`, `Target File Localized`, and `Hit Max Tokens (0%)`.

---

## Hours 4–6: Manuscript Restructuring Checklist (`paper/manuscript.md`)

Edit `paper/manuscript.md` so it reads as a cohesive empirical & systems paper rather than a bug report.

### 3.1 Title, Abstract & Introduction (`paper/manuscript.md` lines 1–50)
- [ ] **Update Title:**
  `When Loss Converges to Zero Capability: Graph-Anchored Verification, Localization Efficiency, and Structural Invariants in Offline Coding Agents`
- [ ] **Rewrite Abstract around the 3 Pillars:**
  1. *Graph-First Localization & Context Efficiency* (>52% token reduction via `get_code_subgraph` + targeted span reads; 85.7%–92.9% file localization on unseen repos).
  2. *Graph-Anchored Verification ($V_{\text{graph}}$ vs. $V_{\text{free}}$)* (+14.3% relative recall gain, $\kappa = 0.7051$ vs. `pytest` $V_{\text{exec}}$).
  3. *The Loss–Execution Paradox* (7 post-training runs converging smoothly to `0.349` loss while end-task resolution remained near zero due to three structural mechanisms invisible to cross-entropy loss: control-token gradient starvation, synthetic schema hallucination, and low-yield RFT overwriting).
- [ ] **Reconcile Run Count in §1:**
  - [ ] Change "five post-training runs" in §1 to **"seven post-training runs (five SFT stages and two RFT passes)"** so §1 matches the 7-run table in §7.2 exactly.

### 3.2 Promote the Empirical Localization & Token Efficiency Analysis (New §4 / §6)
- [ ] **Restore the ATIF-v1.7 Token & Localization Efficiency Table** (from `manuscript_v1_backup.md` lines 375–386, updated with strict submission and localization columns):
  - Show the empirical divergence across the 14 unseen holdout tasks:
    - `Graph-First Targeted Read (1 file)`: **7.5 avg tool calls**, **54,424 avg tokens**.
    - `Moderate Inspection (2–3 files)`: **13.0 avg tool calls**, **108,000 avg tokens**.
    - `Sequential Paging Quicksand (≥4 files)`: **15.0 tool calls (budget exhausted)**, **114,874 avg tokens**.
  - This provides concrete quantitative evidence that graph-guided span targeting compresses token consumption by **>52% in multi-turn rollouts** (>90% per single turn) and prevents turn-budget exhaustion.

### 3.3 Consolidate the "6 Bugs" into 3 Generalizable Post-Training Failure Modes (§5)
Instead of giving equal weight to minor scripting oversights (`try/except: pass` and `or args == {}`), restructure §5 into **three deep technical subsections** plus a brief methodological note on pipeline hygiene:
- [ ] **§5.1 Mechanism 1: Control-Token Gradient Starvation in Folded Chat Templates (formerly F1)**
  - Keep the Appendix A discovery that Gemma 4 folds `<|tool_response>` inside the model turn and omits `<|turn>model` before the final assistant response.
  - Show why token-level state machines starve `<turn|>` of gradient while changing cross-entropy loss by $<0.5\%$, causing run-on generation up to `max_output_tokens`.
- [ ] **§5.2 Mechanism 2: Action-Schema Hallucination in Unverified Teacher Synthesis vs. Grounded Diffs (combines F4 + F6 data findings)**
  - Highlight the corpus sweep table (1,933 tool calls across 5 datasets): **0.0% unexecutable calls in deterministic maintainer-grounded trajectories (`sft_real_train.jsonl`) vs. 27.7%–30.5% unexecutable `edit_file` operations invented by model-generated teacher trajectories** (`insert_after`, multi-replacement arrays) and **7 invented argument names on nullary `submit_patch()`**.
  - Include the `inspect.signature` derivation table across `swegemma.tools`.
- [ ] **§5.3 Mechanism 3: Catastrophic Policy Overwriting in Low-Yield Rejection Fine-Tuning (formerly F3)**
  - Present the within-run loss inversion from `train_log_rft_sft.log`: both RFT passes trained for ~28 seconds on $n=4$ survivors and **doubled the sequence loss** of the SFT checkpoints they overwrote (`0.720 -> 1.414` and `0.638 -> 1.067`) while erasing structured tool calling.
- [ ] **§5.4 Brief Methodological Note: Silent Context Fallbacks & Attempt-Based Metrics (formerly F2, F5, and §5.7)**
  - Condense F2 (`1,110`-token preamble vs. `831`-token median trajectory proof), F5 (attempt vs. resolution metrics), and label-direction check into a concise 1-page subsection on **Measurement & Context Invariants** rather than headlining them as primary research contributions.

### 3.4 Update §6 (Verifier Alignment Study — Table 1)
- [ ] If Path A Step 2A ran: insert the updated `verifier_alignment_table1_v2.json` numbers and the **stratified negative breakdown** (`partial_contract_violation` vs. `sycophantic_drift`), and remove the "provisional" disclaimer.
- [ ] If Path B (no GPU today): keep the existing Table 1 numbers with the transparent threats-to-validity paragraph in §6.

### 3.5 Final Verification & Honesty Gate (30 mins before submission)
- [ ] Run offline test suite to confirm artifact integrity:
  ```bash
  python3 tests/test_chat_format_masking.py
  python3 tests/test_tool_schema_gate.py
  ```
- [ ] Verify every number in `paper/manuscript.md` traces to `experiments/logs/train_log_rft_sft.log`, `experiments/logs/verifier_alignment_table1.json`, or the Cell 8C / Cell 12B outputs.
- [ ] Complete the 5 missing citations in `## References` (`paper/manuscript.md` line 573) using full bibliographic entries.
- [ ] Export/submit `paper/manuscript.md` to the Kaggle Paper Track.
