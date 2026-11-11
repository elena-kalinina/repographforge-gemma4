# Runbook — paper due Nov 12, 2026

**This document is self-contained.** It assumes no memory of how any of it was arrived at.
Everything it asserts is checkable from the repository. Work top to bottom.

---

## 0. State as of end of Nov 10

### Done, verified, committed

- **F6 (interface transcription drift) found, corrected, written up.** Five of the nine
  declared tools diverged from the harness. Corrected in `src/chat_format.py`, in all
  three trajectory generators, and in the data. Full enumeration in manuscript §5.6.
- **Gates 8 and 9 implemented**, with offline tests that need no GPU:
  ```bash
  python3 tests/test_chat_format_masking.py
  python3 tests/test_tool_schema_gate.py
  ```
  Both pass. Run them first on a new machine — they are the fastest proof the checkout
  is sound.
- **Training corpus swept and corrected** (data-prep Cell 8C v3, run successfully).
  Grounded trajectories lost 0 of 114; the two model-generated files lost 27.7% and
  30.5% as unrepairable. Numbers are in §5.4 and §4.2.
- **Master Cells 4, 4B, 0C, 0F run on Kaggle.** 0C reported `outcome=clean`.
  F2 confirmed fixed: the preamble renders at **1,110 tokens** with declarations present.
- **Campaign training log recovered and parsed.** `experiments/logs/train_log_rft_sft.log`
  → `experiments/plots/loss_curves.png` + `experiments/logs/train_loss_points.csv`.
  §7.2 carries the seven-run table.
- **No TODOs remain in `paper/manuscript.md`.**
- README, `PROGRESS_LOG.md` and `competition_plan.md` realigned with the reframed paper.

### Current versions — check these before trusting any Kaggle cell

| Thing | Value |
| :--- | :--- |
| `chat_format.CHAT_FORMAT_VERSION` | `2026-11-11.stage1f` (comment-only change on Nov 11: F2 evidence corrected; schemas unchanged) |
| Tool-schema fingerprint | `649592b1b964a77b` |
| `real_tools.json` generator | `2026-11-10.0F-v2` |

Verify with:
```bash
python3 -c "import sys; sys.path.insert(0,'src'); import chat_format as c; print(c.CHAT_FORMAT_VERSION, c.tool_schema_fingerprint())"
```

### Not done

- Cells 12A/12B (eval-trace forensics) — **written, never run**. This is item B1.
- §2.2 strict submission rates.
- Table 1 re-run.
- The F1 ablation (optional, cuttable).

---

## 1. Setting up a different laptop

```bash
git clone https://github.com/elena-kalinina/Gemma4_Competition.git repographforge-gemma4
cd repographforge-gemma4
python3 tests/test_chat_format_masking.py   # must print ALL CHECKS PASSED
python3 tests/test_tool_schema_gate.py      # must print ALL CHECKS PASSED
```

No GPU, no ADK, no Kaggle session needed for either. If both pass, the checkout is good.

**Python.** The repo's scripts need ≥3.10 (`parse_training_log.py` was 3.12-only until
Nov 11; now 3.9-safe, but do not rely on the macOS system Python). A local `.venv` on
Python 3.12 exists (created Nov 11 with `uv`; `uv` lives at
`~/Library/Python/3.9/bin/uv`, Python 3.12 under `~/.local/share/uv`). Use it:

```bash
source .venv/bin/activate          # or: .venv/bin/python <script>
# recreate from scratch if needed:
~/Library/Python/3.9/bin/uv venv --python 3.12 .venv
~/Library/Python/3.9/bin/uv pip install --python .venv/bin/python matplotlib networkx numpy pandas
```

`matplotlib` is only needed to regenerate the figure; `networkx`/`numpy` for `src/data_loader.py`
and `src/graph_sampler.py`; `pandas` for the Cell 11 audit if run locally.

---

## 2. Where every cell lives

Line numbers are the **body** of each cell — what you paste, excluding the fence.
Re-check them with `grep -n '#### Cell' <file>` if the notebooks have been edited since.

**`notebooks/repographforge_data_preparation.md`**

| Cell | Body lines | Purpose |
| :--- | ---: | :--- |
| 0 | 42–99 | Restart survival check — what `.jsonl` exist, are they clean |
| 8A | 672–785 | Build grounded trajectories (F6-corrected) |
| 8B | 794–828 | Patch-direction gate, 10 samples |
| 8C | 873–1079 | Rewrite existing `.jsonl` to the real tool interface |
| 9 | 1088–1133 | Merge into `sft_combined.jsonl`, report composition |

**`notebooks/repographforge_clean_master.md`**

| Cell | Body lines | Purpose |
| :--- | ---: | :--- |
| 4 | 209–1132 | Write `chat_format.py` — **copy from `src/chat_format.py`, not from here** |
| 4B | 1143–1192 | Version gate on the written formatter |
| 0C | 1219–1478 | Tool-schema diff; writes the training receipt |
| 0D | 1494–1622 | F6 drill-down (diagnostic, already served its purpose) |
| 0E | 1641–1746 | Harness tool surface (diagnostic, already served its purpose) |
| 0F | 1765–1929 | Derive schemas from the harness; writes `real_tools.json` |
| 6 | 1964–2141 | Write `train_sft.py`; calls Gates 8 and 9 |
| 11 | 2271–2427 | Holdout audit with applies-cleanly and RESOLVED columns |
| 12A | 2448–2483 | **Eval-trace schema discovery** |
| 12B | 2495–2577 | **Eval-trace per-task table** |

Every pasteable cell prints a version banner as its first line. **If you do not see the
banner, the paste did not land** and nothing else in that cell's output means what it
appears to. This has caused three silent failures; do not skip the check.

---

## 3. Nov 11 — the work, in order

### B1. Eval-trace forensics — DO FIRST (~45 min, CPU Kaggle session)

The traces from the score-0 run are on Kaggle at `traces/trace_<task>.json`, 14 files for
the 14 holdout tasks, with matching `<task>.log`.

Why this is first: §5.6 argues that mis-declared tool parameters account for "patches
submitted, nothing resolved", but argues it from the schemas and the training data — not
from what the agent did at evaluation time. The traces are what it did.

- [ ] Run **Cell 12A** (master, lines 2448–2483). It dumps the schema of one trace.
- [ ] Read the schema. 12B probes generically and its numbers should not be trusted until
      the layout is confirmed. If 12A shows something 12B will not find, fix 12B first.
- [ ] Run **Cell 12B** (lines 2495–2577). The line that matters:
      ```
      submitted with NO edit : n/14   <-- F6's predicted signature
      ```
- [ ] If that count is high, add it to §5.6 and let §7 state that F6 **accounts for** the
      score-0 submission pattern. Keep "accounts for" — no corrected run exists, so
      "caused" is not available.
- [ ] If 12B reports traces it could not parse, fix the extraction rather than reporting
      the partial number. Reporting a confident count over an unknown denominator is
      failure F5, which this table exists to correct.

### B2. Strict submission rates (~1–2 h)

- [ ] Recompute historical submission rates under the strict metric: **applies cleanly
      AND tests pass**, not attempts. This is the honest version of the 85.7% → 0.0%
      claim in §5.5 and it needs a table.
- [ ] Source: the same Kaggle eval outputs as B1, plus `<task>.log`.

### B3. Table 1 — the one positive result (~2 h, needs GPU)

Current numbers in §6, from `experiments/logs/verifier_alignment_table1.json`:
`V_free` P 0.8660 / R 0.6512 / F1 0.7434 / κ 0.6401 against
`V_graph` P 0.8571 / R 0.7442 / F1 0.7967 / κ 0.7051.

- [ ] Re-run with task-specific negatives (`src/candidate_patch_generator.py`).
- [ ] Confirm the judge is the **base** model, never `main_lora`.
- [ ] Report stratified by negative class.
- [ ] State n, how negatives were generated, and that they are synthetic.
- [ ] **If it does not re-run in time:** mark Table 1 provisional in the caption and say
      so in §9. It is already marked provisional in §6 — leave that marking in place.
      Do not ship it unmarked.

### B4. OPTIONAL — the F1 ablation

- [ ] Two adapters identical but for stop-token masking; report ceiling-hit rate and
      submission rate.
- [ ] **Cut this first if time runs short.** §9 already labels F1 observational, which is
      defensible. A rushed ablation is not.

### B5. The honesty pass (~1 h) — NEVER CUT

- [ ] Every number in the paper traces to a file in `experiments/logs/`. Anything that
      does not, delete. (Checked once on Nov 10 for the training-loss numbers — all of
      them trace. Table 1 and the submission rates have not been checked this way.)
- [ ] Read start to finish as a hostile reviewer, asking of each claim: *what measurement
      supports this?*
- [ ] **Open inconsistency, unresolved:** §1 says "five post-training runs" and then lists
      six leaderboard scores (0.00, 0.00, 0.00, 0.03, 0.01, 0.00). The training log shows
      **seven** training runs, two of which were RFT passes. Decide what "run" means in
      §1 and make the two consistent. This needs your knowledge of which submission was
      which; it cannot be settled from the repository.
- [ ] Confirm §5.6's second-order admission survives editing — the `list[str]` mistyping
      in our own schema-derivation script. It is the most credible paragraph in the
      section and the easiest to cut by accident.

---

## 4. Nov 12 — submit

- [ ] Artifact release: `src/chat_format.py`, `src/parse_training_log.py`,
      `tests/test_chat_format_masking.py`, `tests/test_tool_schema_gate.py`,
      master Cells 0C–0F and 12A/12B, data-prep Cell 8C. The gates are the contribution;
      ship them runnable.
- [ ] Final read.
- [ ] Submit.

---

## 5. Resuming for training (Dec 2 track — NOT for Nov 11)

`/kaggle/working` has survived session restarts, so the artefacts persist: `chat_format.py`,
`real_tools.json`, the receipt, `train_sft.py`, the corrected `.jsonl`. What does **not**
persist is anything installed into the Python environment or defined in memory.

| Cell | Re-run? | Why |
| :--- | :--- | :--- |
| Master 1, 2, 3 | **Always** | `pip install` and module globals die with the session |
| Master 4 | **Yes** | `chat_format` is `stage1f`; also re-establishes the module |
| Master 4B | **Always** | It is the gate. Cheap, and it reports drift |
| Master 0C–0F | **No**, if 4B passes | The receipt persists and the gate validates it |
| Data prep 8A–9 | **No** | The `.jsonl` persist and are F6-corrected |
| Master 5, 6, 7 | **Yes** | GPU flush, rewrite `train_sft.py`, train |

**You do not have to remember this.** `train_sft.py` calls `assert_tool_schema_verified()`
and `assert_trajectories_conform()` before the model loads. A missing, stale or mismatched
receipt refuses with the remedy named. The gate is the memory.

One exception: `chat_format` moved to `stage1f` after the last Kaggle run, which
invalidates the existing receipt by version. **Before the next training run:**
4 → 4B → 0F → 0C → 6, once. That is the version gate working, not a setback.

Remaining training work is in `TRAINING_FIX_PLAN_DEC2.md`. Its first open decision:
`sft_combined.jsonl` was cleaned in place (304 → 250) rather than re-merged by Cell 9.
Re-running Cell 9 rebuilds it from the cleaned sources (114 grounded + 82 synthetic),
giving a different count and a majority-grounded ratio. Both files are conformant. Pick
one, record which, and report its composition in §4.2.

---

## 6. Findings a cold reader needs

These are argued in full in `paper/manuscript.md`; this is the index.

| Finding | Where |
| :--- | :--- |
| F6: five of nine tools mis-declared; `submit_patch` takes **no arguments** while every trajectory sent one | §5.6 |
| F2: declarations absent by construction in runs 1–5; fail-open path in runs 6–7 **undetermined**. The "median 831 < preamble 1,110" argument was retracted 2026-11-11 (median was raw-JSON, preamble-free). Corrected renderer measures 1,110 tokens | §5.2, §4.4, §9.11 |
| 26.0% of 1,933 tool calls were string-encoded; the rest dicts, same corpus | §5.4 |
| Model-generated trajectories invented four `edit_file` schemas; ~30% unrepairable | §5.4, §4.2 |
| **Two** RFT passes, both on 4 examples for ~28 s, both ending at ~2× the loss they overwrote | §5.3, §7.2 |
| The best-converged run (0.349, lowest of the campaign) scored **0.00** | §7.2 |
| Run-to-submission mapping deliberately **not** inferred | §7.2, §9.9 |

---

## 7. Documents

| File | Contents |
| :--- | :--- |
| `paper/manuscript.md` | The paper |
| `PAPER_FINALIZATION.md` | Full checklist with per-item history |
| `TRAINING_FIX_PLAN_DEC2.md` | Dec 2 training work |
| `STATUS_REVIEW.md` | Post-fix audit, including a retracted section |
| `DIAGNOSIS.md` | Original 13-defect diagnosis |
| `PROGRESS_LOG.md`, `competition_plan.md` | Pre-reframe logs, with retraction banners |
