# RepoGraphForge: Fixed Steps & Remediation Checklist (`FIXED_STEPS_PLAN.md`)

**Companion to:** `competition_plan.md`, `DIAGNOSIS.md`, and `FIX_PLAN.md`  
**Paper Track Deadline:** Thursday, November 12, 2026  
**Code Competition Deadline:** December 2, 2026  

---

## 1. Root-Cause Remediation Matrix (Code & Notebook Implementation Status)

All 13 root causes identified in `DIAGNOSIS.md` (`A1–A3`, `B1–B5`, `C1–C5`) have been implemented in the repository (`src/` and `notebooks/`):

| ID | Defect Summary | Implemented Fix | Target Files | Status |
| :--- | :--- | :--- | :--- | :---: |
| **A1** | SFT label used `bug_patch` (bug-introducing diff) instead of `fix_patch`; 115 real `train_tasks.jsonl` gold fixes were unused. | Built `src/build_real_sft.py` (Data Prep **Cell 8A**) to construct grounded trajectories from the 115 real `train_tasks.jsonl` `patch` (`fix_patch`, asserting `"fix" in field_name`), plus inverted `fix_patch` in `src/task_synthesizer.py` & `src/teacher_rollouts.py`. | `src/build_real_sft.py`, `src/task_synthesizer.py`, `src/teacher_rollouts.py`, `notebooks/repographforge_data_preparation.md` | ✅ Code Ready |
| **A2** | Synthesizer prompted with node names only, inventing context lines that cannot `git apply`. | Grounded real-task builder uses actual hunks from `train_tasks.jsonl`; synthetic tasks include `node["text"]` and `start_line`/`end_line` spans. | `src/build_real_sft.py`, `src/task_synthesizer.py` | ✅ Code Ready |
| **A3** | Teacher trajectories were unverified LLM screenplays (`len(msgs) >= 4`); `reproduction_test` was dropped in Cell 7. | Preserved `reproduction_test` in Cell 7; grounded `read_file` observations contain `old_string` verbatim; added Phase 2.2 direction check (**Cell 8B**). | `src/build_real_sft.py`, `notebooks/repographforge_data_preparation.md` | ✅ Code Ready |
| **B1** | End-of-turn `<turn\|>` token was masked (`-100`); `<\|turn>` matched user turns too, so model never learned to stop and generated `<\|turn>user`. | Implemented `build_example()` and `_mask_rendered_token_ids()` in `src/chat_format.py` to unmask assistant spans **inclusive of `<turn\|>`** while masking user turns, `<\|turn>model\n` headers, and `<\|tool_response>` blocks. Added `verify_and_print_samples()` gate. | `src/chat_format.py`, `src/train_qlora.py`, `notebooks/repographforge_clean_master.md` (Cells 4 & 6) | ✅ Code Ready |
| **B2** | `MAX_LEN = 1536` right-truncated `edit_file` and `submit_patch` off training sequences. | Raised `MAX_LEN = 8192` everywhere and enforced a strict **drop-don't-truncate** policy (`if len(input_ids) > max_len: return None`), logging dropped counts. | `src/chat_format.py`, `src/train_qlora.py`, `notebooks/repographforge_clean_master.md` | ✅ Code Ready |
| **B3** | RFT trained on 4 markdown-fence examples with no masking and self-judged by `main_lora`. | Defaulted Phase 4 to **SFT-only** (`lr=1e-4`, 2 epochs); updated `src/rft_sampler.py` to extract diffs from `edit_file` tool calls, judge with the **base model**, and require `--min-accepted 50` in `src/train_rft.py`. | `src/rft_sampler.py`, `src/train_rft.py`, `notebooks/repographforge_clean_master.md` | ✅ Code Ready |
| **B4** | Double-BOS (`apply_chat_template` added `<bos>`, then `tokenizer()` added a second `<bos>`). | Tokenized rendered strings with `add_special_tokens=False`, stripped any duplicate leading `<bos>`, and asserted `input_ids.count(bos_id) == 1`. | `src/chat_format.py` | ✅ Code Ready |
| **B5 & C1** | Four incompatible tool-call serializations and divergent trainers across `src/` and `notebooks/`. | Unified all rendering, normalization, and tokenization into `src/chat_format.py`, imported by `train_qlora.py`, `train_rft.py`, and Master Notebook **Cell 4/6**. | `src/chat_format.py`, `src/train_qlora.py`, `src/train_rft.py` | ✅ Code Ready |
| **C2** | Training sequences had no system prompt and no `tools=` declarations, unlike eval time. | `chat_format.py` injects `DEFAULT_SYSTEM_PROMPT` and all 9 `COMPETITION_TOOLS` schemas into `apply_chat_template`. | `src/chat_format.py` | ✅ Code Ready |
| **C3** | `max_output_tokens: 1024` truncated multi-hunk `edit_file` / `submit_patch` calls; dead config in `submission/agent.yaml`. | Raised `max_output_tokens: 4096` in `configs/sampling.yaml` (**Cell 8**) and synced `submission/agent.yaml` to the official ADK schema. | `notebooks/repographforge_clean_master.md` (Cell 8), `submission/agent.yaml` | ✅ Code Ready |
| **C4** | Anti-hesitation prompt forbade reading $>1$ file or running `pytest`. | Replaced with the Phase 3.4 budget prompt (up to 3 file reads, `search_similar_code` fallback, `run_command` / `pytest` verification) and added `EVAL_ARM = "A" \| "B" \| "C"` switch in **Cell 8**. | `src/chat_format.py`, `notebooks/repographforge_clean_master.md` (Cell 8) | ✅ Code Ready |
| **C5** | Audit table (`or args == {}`) counted zero-arg tools (`get_status()`) as `submit_patch`. | Removed `or args == {}` from **Cell 11**, checking strictly `name == "submit_patch"` and non-empty diff length, plus logging `Hit Max Tokens`. | `notebooks/repographforge_clean_master.md` (Cell 11) | ✅ Code Ready |

---

## 2. Step-by-Step Execution Checklist (Tick Off as You Run in Kaggle)

### Phase 0: Establish Ground Truth & Snapshot Baseline (100% CPU — 0 GPU Quota)
*Notebook: `notebooks/repographforge_clean_master.md` (Part 0)*

- [x] **0.0 Code Implementation:** Add Phase 0 diagnostic and backup cells (`Cell 0A` and `Cell 0B`) to `notebooks/repographforge_clean_master.md`.
- [ ] **0.1 Dump Real Chat Template & BOS Behavior (`Cell 0A`):**
  - Run `Cell 0A` on Kaggle CPU to print the rendered Gemma 4 chat template, turn markers (`<|turn>model`, `<turn|>`), and verify whether `add_special_tokens=True` duplicates `<bos>`.
- [ ] **0.2 Confirm vLLM Tool-Call Response Structure:**
  - Verify that native `<|tool_call>call:name{...}<tool_call|>` / `tool_calls` dicts match the ADK parser.
- [ ] **0.3 Record Trajectory Token Length Distribution (`Cell 0A`):**
  - Log `[50, 90, 95, 99]` token length percentiles and record the exact fraction of trajectories previously truncated at `MAX_LEN = 1536` for the paper.
- [ ] **0.4 Snapshot Current Baseline Adapter (`Cell 0B`):**
  - Run `Cell 0B` to copy `/kaggle/working/submission/adapters/main_lora` $\rightarrow$ `/kaggle/working/experiments/adapters/rft_v1_baseline` so Control Arm A is preserved.

---

### Phase 1: Canonical Formatting & Loss-Masking Gate (100% CPU — 0 GPU Quota)
*Files: `src/chat_format.py`, `src/train_qlora.py`, `notebooks/repographforge_clean_master.md` (Cell 4 & Cell 6)*

- [x] **1.1 Turn-by-Turn & Span Assistant Masking (`B1`):** Implement `build_example()` in `src/chat_format.py` so assistant turns + closing `<turn|>` carry gradient while user turns, `<|turn>model\n` headers, and `<|tool_response>` observations are masked with `-100`.
- [x] **1.2 Pre-Training Assertions & Decoded Span Printer (`B1`):** Implement `verify_and_print_samples()` in `src/chat_format.py` (asserts non-empty labels, masked user turns, single BOS, and trainable EOT token, and prints decoded trainable text on 3 samples).
- [x] **1.3 `MAX_LEN = 8192` Drop-Don't-Truncate Policy (`B2`):** Drop overflowing examples (`return None`) instead of right-truncating `edit_file` / `submit_patch`, and log dropped count.
- [x] **1.4 Single-BOS Hygiene (`B4`):** Tokenize rendered strings with `add_special_tokens=False` and assert `input_ids.count(bos_id) == 1`.
- [x] **1.5 Unified Formatter Module (`B5` / `C1`):** Consolidate all tokenization into `src/chat_format.py` (`Cell 4` in Master Notebook) and import into `train_sft.py` (`Cell 6`).
- [ ] **1.6 Execute `Cell 4` in Kaggle:** Write `/kaggle/working/chat_format.py` and confirm local smoke test passes.

---

### Phase 2: Grounded Real-Task SFT Dataset & Patch Direction Gate (100% CPU — 0 GPU Quota)
*Files: `src/build_real_sft.py`, `notebooks/repographforge_data_preparation.md` (Cells 8A, 8B, 9)*

- [x] **2.1a Explicit `fix_patch` Inversion & Naming (`A1`):** Update `src/task_synthesizer.py` and `src/teacher_rollouts.py` to output `fix_patch` (`assert "fix" in field_name`) and preserve `reproduction_test`.
- [x] **2.1b Grounded SFT Builder from 115 Real Training Tasks (`A1` / `A2`):** Implement `src/build_real_sft.py` and Data Prep **Cell 8A** to build grounded trajectories from the 115 real `train_tasks.jsonl` (`fastapi` and `rich`) using their verified maintainer `patch` (`fix_patch`).
- [ ] **2.2 Run `Cell 8A`, `Cell 8B`, and `Cell 9` in Kaggle (CPU):**
  - [ ] Run **Cell 8A** to generate `/kaggle/working/data/trajectories/sft_real_train.jsonl` (115 real tasks).
  - [ ] Run **Cell 8B** to verify 10/10 forward patch directions (`old_string` in `read_file`, `new_string` matches `+` additions in `fix_patch`).
  - [ ] Run **Cell 9** to merge `sft_real_train.jsonl` into `/kaggle/working/data/trajectories/sft_combined.jsonl`.

---

### Phase 3: Train/Eval Contract Alignment (100% CPU — 0 GPU Quota)
*Files: `src/chat_format.py`, `submission/agent.yaml`, `notebooks/repographforge_clean_master.md` (Cells 8 & 11)*

- [x] **3.1 Single Tool-Call Serializer (`C1`):** All trajectories normalized with `dict` arguments and `name`/`tool_call_id` on `role == "tool"` messages via `chat_format.py`.
- [x] **3.2 Eval-Time System Prompt & 9 Tool Schemas in Training (`C2`):** `COMPETITION_TOOLS` and `DEFAULT_SYSTEM_PROMPT` passed into `apply_chat_template` during training.
- [x] **3.3 Raise Output Token Cap to `4096` (`C3`):** Updated `configs/sampling.yaml` (`Cell 8`) to `max_output_tokens: 4096` and aligned `submission/agent.yaml`.
- [x] **3.4 Budget-Based System Prompt (`C4`):** Replaced `MAX 1 FILE READ` prohibition with the 15-call budget workflow (up to 3 file reads, `search_similar_code` fallback, `run_command` / `pytest` verification before `submit_patch`).
- [x] **3.5 Fix Audit Table Submission Metric (`C5`):** Removed `or args == {}` from `Cell 11` in `notebooks/repographforge_clean_master.md` and added `Hit Max Tokens` counter.

---

### Phase 4: Retrain SFT Only & Run 3-Arm Holdout Evaluation (GPU L4 x4 — ~1–2 GPU Hrs)
*Notebook: `notebooks/repographforge_clean_master.md` (Parts 2, 3, 4)*

- [ ] **4.1 Launch Clean SFT Training (`Cells 5, 6, 7` — No RFT Overwrite):**
  - [ ] Run `Cell 5` (flush GPU VRAM) and `Cell 6` (write `/kaggle/working/train_sft.py` with `MAX_LEN=8192`, `lr=1e-4`, `epochs=2`).
  - [ ] Run `Cell 7` (`!python3 /kaggle/working/train_sft.py`) and **inspect the printed `PHASE 1.2 PRE-TRAINING MASKING & EOT VERIFICATION GATE` output** on the 3 samples before the 31B model loads.
  - [ ] Confirm training completes 2 epochs and exports sanitized adapter to `/kaggle/working/submission/adapters/main_lora`.
- [ ] **4.2 Run 3-Arm Holdout Evaluation on the 14 OOD Tasks (`val_ood_tasks.jsonl`):**
  - [ ] **Arm C (`EVAL_ARM = "C"`, Recommended Primary Run):** New SFT adapter + Phase 3.4 Budget Prompt (`Cell 8` $\rightarrow$ `Cell 9` $\rightarrow$ `Cell 10` $\rightarrow$ `Cell 11`). Record strict `submit_patch` rate, non-empty diff rate, `pytest` `resolved=True` rate, and `Hit Max Tokens` count.
  - [ ] **Arm B (`EVAL_ARM = "B"`, Optional Ablation):** New SFT adapter + legacy anti-hesitation prompt (isolates training fixes `B1/A1/B2/C2` vs. prompt change `C4`).
  - [ ] **Arm A (`EVAL_ARM = "A"`, Optional Control):** Baseline `rft_v1_baseline` snapshot + legacy prompt.
- [ ] **4.3 Package & Submit Best Arm to Leaderboard (`Cells 12 & 13`):**
  - [ ] Run Section 6 (`Cell 12`) to package `submission.zip` and submit to the Kaggle leaderboard.

---

### Phase 5 & Phase 6: Stretch Goals (Only If Quota & Time Allow After Phase 4)
- [ ] **5.1 Grounded Synthetic Filtering (`git apply --check`):** Filter `synthetic_tasks.jsonl` against repository checkouts in `snapshots/`.
- [ ] **5.2 Execution-Verified Sandbox Rollouts:** Collect rollouts directly from the live `swegemma` sandbox where `submit_patch` is called and `pytest` exits with `0`.
- [ ] **6.1 Gated RFT Rebuild (`src/rft_sampler.py` & `src/train_rft.py`):** Run RFT only after accumulating $\ge 50$ verified tool-calling trajectories, judged by the unadapted **base** model (`--min-accepted 50`).

---

### Phase 7: Manuscript & Benchmark Integrity Corrections (100% CPU — Mandatory Before Nov 12)
*Files: `src/candidate_patch_generator.py`, `notebooks/repographforge_data_preparation.md` (Cells 4 & 5), `paper/manuscript.md`, `PROGRESS_LOG.md`*

- [x] **7.1 Clarify Grounded vs. Synthetic Trajectories:** Updated `paper/manuscript.md` and `PROGRESS_LOG.md` to distinguish synthetic teacher rollouts from `train_tasks.jsonl` grounded trajectories (`src/build_real_sft.py`).
- [x] **7.2 Replace Constant `fix.py` Negative in Table 1:** Updated `src/candidate_patch_generator.py` and Data Prep **Cell 4 & Cell 5** to generate task-specific sycophantic drift patches on the real target file/hunk and report false-positive rates stratified by negative class (`partial_contract_omission` vs. `sycophantic_drift`).
- [x] **7.3 Uncontaminated Base-Model Judge:** Updated Data Prep **Cell 5** and `src/rft_sampler.py` to query the base model (`gemma-4-31b-it-qat-w4a16-ct`), never `main_lora`.
- [x] **7.4 Document Corrected Submission Metric (`C5`) & `B1` EOT Ablation:** Updated `paper/manuscript.md` Section 5.2 & 5.3 and `PROGRESS_LOG.md` with the `B1` end-of-turn masking diagnosis, `B3` RFT markdown-fence finding, and `C5` strict submission metric.
- [ ] **7.5 Final Table 1 & Table 2 Update After Phase 4 Run:**
  - [ ] Re-run Data Prep **Cell 4 & Cell 5** (if updating Table 1 with task-specific negatives) and insert final Phase 4.2 Arm B/C evaluation metrics into `paper/manuscript.md` before submitting on November 12.
