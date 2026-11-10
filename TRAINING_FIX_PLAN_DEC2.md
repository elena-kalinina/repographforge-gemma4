# Training Fix Plan — Nov 13 → Dec 2 (Code Competition)

Goal: **a non-zero leaderboard score**, via 2–3 deliberate submissions rather than many
blind ones. Prerequisite: the paper is submitted Nov 12 and is not competing for attention.

**Operating rule, learned the expensive way:** every GPU run must be preceded by a CPU
check that would have caught the last failure. Four runs have now been spent on defects
that a five-minute CPU assertion would have surfaced. Budget the checks, not the runs.

**Standing rule:** no run starts until every box in Stage 1 is ticked.

---

## Stage 1 — Close the verification gaps (CPU only, ~1 day, Nov 13)

> **Status 2026-11-10:** everything implementable offline is done and committed.
> `[x]` = implemented and syntax-checked locally. `[~]` = code written (Cell 0C), must be
> RUN on Kaggle where the ADK is importable. The masking fix is additionally verified by
> `tests/test_chat_format_masking.py`, which replays the real Phase 0.1 template through
> `_mask_rendered_token_ids` and asserts the stop token is supervised while user turns and
> folded tool observations are not. It passes.

### 1.1 Tool schemas — the top suspect for the score-0 run
- [x] Cell 0C v3 written. **v1 post-mortem:** it probed six guessed global names
      (`TOOLS`, `tools`, …) on the assumption that Ryan's Section 3 defines the tools as
      notebook variables. It does not — the harness tools live inside the installed ADK
      packages — so discovery could never have succeeded. Worse, the F2 render probe was
      nested inside the discovery `else`, so the miss silently skipped the one check that
      needed no discovery. The run produced no evidence at all.
- [x] v2/v3 discovery: structural search over globals, a walk of
      `swegemma`/`adk_eval_core`/`adk_submission`, and a **source-level grep** of the
      harness `.py` files for all 9 tool names and 6 critical parameter names. The grep
      runs unconditionally — a tool name absent from every harness source file is scored
      as a mismatch on its own, with no importable registry required.
- [x] F2 probe hoisted to top level: it always runs and always reports the preamble token
      count (manuscript §5.2 evidence).
- [x] **Hard gate** (`chat_format.assert_tool_schema_verified`). Cell 0C writes a receipt
      carrying the chat_format version, a sha256 fingerprint of the schemas, the outcome
      and the F2 result. `train_sft.py` calls the gate before the model loads and refuses
      on: no receipt, stale version, schemas edited since verification, failed F2,
      recorded mismatch, or an inconclusive verification without an explicit
      `TOOL_SCHEMA_ACK=1`. An acknowledgement cannot rescue a real mismatch.
      Covered by `tests/test_tool_schema_gate.py` (10 checks, offline).
- [x] **RUN Cell 0C on Kaggle** (2026-11-10). Results:
      - All 9 tool **names** present in the harness source (98 .py files scanned).
      - `node_id` — the *required* parameter of `get_code_subgraph` and
        `get_code_neighbors` — appears in **zero** harness .py files. Every other
        required parameter we declare (`filepath`, `old_string`, `new_string`,
        `command`, `patch`) is present. This is the live suspect.
      - No importable tool registry found → `outcome=inconclusive`; the gate will
        refuse training until Cell 0D resolves `node_id` or `TOOL_SCHEMA_ACK=1` is set.
      - **F2 is FIXED**: preamble renders at **1,110 tokens**, declarations present.
      - `submit_patch_tool` / `edit_file_tool` in the undeclared list confirm the
        harness wraps tools as `<name>_tool`, which Cell 0D exploits.
- [ ] **RUN Cell 0D** to settle `node_id`. Blocking.
- [ ] Record every mismatch found. If any parameter name differed, that is very likely the
      whole explanation for "6/14 submitted, score 0" — `edit_file` rejecting the call,
      then `submit_patch` on an unmodified tree.
- [ ] If the diff comes back clean, the score-0 cause moves to patch application /
      `submit_patch` semantics — see the `git apply --check` column added to Cell 11.

### 1.2 Make silent fallbacks loud
- [x] `chat_format.py:render` and `build_example` both wrap the `tools=` render in
      `try/except Exception: pass`. Make them **raise**, or log at ERROR and count.
- [x] Assert the rendered training text contains the tool declarations:
      `assert "old_string" in rendered and "node_id" in rendered`.

### 1.3 Tighten the masking gate
- [x] Log which masking path ran (`prefix_ok` True/False) and the counts. The fallback
      path can unmask tool observations entirely if the template renders `tool` as its own
      turn role — `_mask_rendered_token_ids` only matches `system`/`user`/`model`.
- [x] Add to `verify_and_print_samples`:
      ```python
      frac = len(unmasked_ids) / len(ids)
      assert 0.03 < frac < 0.45, f"Unmasked fraction {frac:.1%} out of range"
      for marker in ("File: ", '"status": "ok"', "### Problem", "<|tool_response>"):
          assert marker not in tokenizer.decode(unmasked_ids), f"Observation leaked: {marker}"
      ```
- [ ] Read the decoded output for 3 samples yourself. Not optional.

### 1.4 Measure what you're training on
- [x] Report the composition split: 115 grounded vs ~184 synthetic screenplays (62%).
- [x] Fix the merge bug in Data Prep Cell 9 — the loop does
      `all_trajectories[syn_id] = item` unconditionally, so `sft_scaled` **overwrites**
      `sft_real_train` on any key collision, the opposite of the "real tasks first"
      comment. Use `if syn_id not in all_trajectories`.
- [x] Add a floor: `assert len(kept) >= 60`. Scale `gradient_accumulation_steps` so you
      get **≥30 optimizer steps**, not 2.

### 1.5 Make the eval falsifiable
- [x] Add **applies-cleanly** (`git apply --check`) and **tests-pass** columns to the audit.
- [ ] Until these exist you cannot distinguish a good checkpoint from a bad one, and every
      local number is unfalsifiable. This is the single highest-value item in Stage 1.

---

## Stage 2 — Memory fix, restore real context (CPU to write, ~0.5 day, Nov 14)

### 2.1 Chunked masked loss
- [ ] Implement the `MemEfficientTrainer.compute_loss` from `STATUS_REVIEW.md` §1: gather
      supervised positions, then chunk `lm_head`+CE under `checkpoint`.
- [ ] Verify with `torch.cuda.max_memory_allocated()` on **one** step before a full run.
- [ ] Expected peak logits at MAX_LEN=8192: ~0.54 GB (vs ~8.6 GB full-vocab fp32).

### 2.2 Restore MAX_LEN
- [ ] Raise to 8192 once 2.1 verifies. Re-measure kept/dropped — with only 5 dropped at
      2560, raising the cap mostly buys headroom for the preamble and for longer grounded
      trajectories, which is what you want.
- [ ] If memory still binds: **left-truncate** rather than drop, so `edit_file` and
      `submit_patch` always survive at the tail.

---

## Stage 3 — Submission 1: clean SFT (GPU ~1.5h, Nov 15–16)

- [ ] Train SFT only. No RFT. `lr=1e-4`, 2 epochs, MAX_LEN per Stage 2.
- [ ] **Read the Phase 1.2 gate output before the 31B loads.** Kill the run if it looks wrong.
- [ ] Evaluate on the 14 OOD tasks with the Stage 1.5 metrics.
- [ ] Record: applies-cleanly, tests-pass, strict submit rate, % hitting token ceiling,
      mean tool calls.
- [ ] Compare against the `rft_v1_baseline` snapshot on the same tasks.
- [ ] **Submit if and only if tests-pass > 0 locally.** Do not spend a submission to find
      out what a local run would have told you.

### Decision gate
- **tests-pass > 0** → Stage 4.
- **Submits but nothing applies** → tool-call schema or `edit_file` argument problem.
  Return to 1.1; do not retrain.
- **Still no submissions** → masking or stop token. Return to 1.3; do not retrain.

---

## Stage 4 — Submission 2: grounded data only (GPU ~2h, Nov 18–21)

The 62% synthetic share is the largest remaining data-quality problem: those trajectories
have hallucinated observations and were never execution-verified.

- [ ] Train on the **115 grounded trajectories only**. Compare to Stage 3.
      This is a clean, interpretable ablation and it's also a paper-worthy result.
- [ ] If grounded-only wins, drop the synthetic set.
- [ ] Expand grounded data (`FIX_PLAN.md` Phase 5): put real source spans in the
      synthesizer prompt, `git apply --check` every generated patch, discard failures.
- [ ] Make trajectory observations match **real tool output format**. `build_real_sft.py`
      currently invents them (`f"File: {filepath} (lines x-y)\n{numbered}"`,
      `json.dumps(subgraph_payload)`). Capture one real observation of each type from the
      sandbox and mirror it exactly.
- [ ] Retrain, evaluate, submit if it beats Stage 3 locally.

---

## Stage 5 — Submission 3, optional (GPU ~2h, Nov 24–28)

Only if Stages 3–4 produced a non-zero score. Pick **one**:

- [ ] **(a) Execution-verified rollouts.** Run the real agent loop in the sandbox; keep
      only rollouts where `submit_patch` fired, the diff applied, and tests flipped
      red→green. The highest-value remaining work.
- [ ] **(b) Gated RFT rebuild.** Only with ≥50 accepted tool-calling trajectories, base
      model as judge, masking per Stage 1.3. Do not resurrect the old RFT path.
- [ ] **(c) Best-of-N at inference.** No training. Sample N=4 at temperature, keep the
      candidate whose diff applies and passes tests. Often the cheapest real gain, and it
      sidesteps training entirely.

**(c) first** if quota is tight — it needs no training run and can be tested locally.

---

## Stage 6 — Final submission hygiene (Dec 1–2)

- [ ] Resolve which `agent.yaml` the grader reads — `submission/agent.yaml` and the one
      Cell 8 writes are still in different schemas. **Delete the dead one.**
- [ ] Confirm `max_output_tokens: 4096` in the packaged artifact.
- [ ] Validate `submission.zip` with the ADK validator.
- [ ] Verify the packaged adapter is the checkpoint you actually chose — `train_rft.py`
      writes to the same path it reads from, which is how the SFT v2 checkpoint was lost.
- [ ] Keep one submission slot in reserve.

---

## Budget

| Stage | GPU | Calendar |
|---|---|---|
| 1 Verification gaps | 0 | Nov 13 |
| 2 Memory fix | 0 | Nov 14 |
| 3 Clean SFT + submission 1 | ~1.5 h | Nov 15–16 |
| 4 Grounded-only + submission 2 | ~2 h | Nov 18–21 |
| 5 Optional third | ~2 h | Nov 24–28 |
| 6 Packaging | 0 | Dec 1–2 |

~5.5 GPU hours across three weeks, most of the work on CPU. If you only get through
Stages 1–3, you will still have learned more than the last four runs combined, because
for the first time the eval will be able to tell you whether a patch actually works.
