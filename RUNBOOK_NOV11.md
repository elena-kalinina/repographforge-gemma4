# Runbook — Nov 11, one session, paper due Nov 12

Written 2026-11-10 late. Everything here is ordered. Do it top to bottom.

`chat_format` is at **2026-11-10.stage1d**. Anything pasted into Kaggle before tonight is
stale. The gates will tell you, but re-pasting first is faster than being told.

**Time budget.** Part A is ~30 min of CPU session. Part B is the paper and is the whole
day. Part C is Nov 12. If Part B runs long, cut from §B4 (marked OPTIONAL), never from §B5.

---

## Part A — Kaggle CPU session: land the F6 correction (~30 min)

F6: five of nine declared tools diverged from the harness. Corrected in the repo; none of
it is in your Kaggle notebooks yet.

### A0. After a session restart — what you do and do not need

**Data preparation notebook: no prerequisites.** Cell 8A imports only `json` and `re`,
creates its own directories, and auto-partitions `train_tasks.jsonl` from the competition
input when it is missing. It needs no ADK, no transformers patch, no graph data, no GPU —
so Ryan's Sections 1–3 are **not** required here. Running data-prep Cell 1 first is
harmless and costs seconds, but 8A does not depend on it.

- [ ] **Cell 0 (NEW)** — run first. Reports which `.jsonl` survived the restart, whether
      each still contains stale arguments, and whether `real_tools.json`, `chat_format.py`
      and the receipt are present.

**The one thing that may not survive:** `sft_scaled.jsonl` is produced by
`src/teacher_rollouts.py`, a GPU/LLM job that is **not** in either notebook. If `/kaggle/
working` was cleared, 8A rebuilds the grounded 115 but nothing rebuilds the synthetic
~184, and Cell 8C will have only the grounded file to rewrite. For the paper this blocks
nothing — run the sequence on what exists and state the subset. For Dec 2 it becomes a
choice between regenerating the synthetic set and training grounded-only, and
grounded-only was already Stage 4's ablation.

**Master notebook: yes, you need Ryan's Sections 1–3 and the transformers fix.** Unlike
data prep, these are real dependencies:

- [ ] **Cell 1** — installs `swegemma` / `adk_eval_core` / `adk_submission` from the
      wheelhouse. Cell 0F imports `swegemma.tools`; without this it cannot run at all.
- [ ] **Cell 2** — transformers 5.13.1 decorator patch and torchao disarm. Cells 0C and 0F
      both construct an `AutoTokenizer`, which fails without it.
- [ ] **Cell 3** — Ryan's Sections 2 & 3, paths and graph utilities. Cells 0C/0F do not
      strictly need it; Cell 6 does, and it is cheap. Run it.

Only then 4 → 4B → 0F → 0C → 6.

### A1. Data preparation notebook

Run order is **Cell 0 → 8A → 8C → 8B → 9**, which is not the notebook's own order. 8B is a gate and
should see the final artefact, not an intermediate. 8C does not touch `edit_file` or
`read_file`, which is all 8B inspects, so this reordering cannot change 8B's verdict — it
just means the verdict is about the file you will actually train on.

- [ ] **Cell 8A** — re-paste, run. Emits `{"nodes": [anchor]}` and `submit_patch` with no
      arguments. ~5 s on CPU. Regenerates `sft_real_train.jsonl`.
- [ ] **Cell 8C** — NEW, paste, run. Rewrites every `.jsonl` in place for the tools 8A
      does not regenerate — chiefly `sft_scaled.jsonl`, whose generator is not in this
      notebook. Backs each file up to `.pre_f6`. Idempotent.
      Expect: a per-file count, the number of string-encoded calls, a breakdown of
      rewrites, and the distinct `submit_patch` argument names. Ends `✓ All trajectories
      conform to the real tool interface.` If it reports offenders instead, it now names
      file, line, tool and raw arguments — paste that rather than the traceback.
- [ ] **Capture the 8C v2 output verbatim.** Two numbers go straight into the paper:
      the string-encoded call count (§5.4) and the full `submit_patch` argument-name
      table (§5.4 has a TODO holding the v1 lower bound of seven names).
- [ ] **Cell 8B** — run (unchanged). 10/10 must pass.
- [ ] **Cell 9** — re-paste if you have not since the merge-bug fix, run. Check the
      grounded/synthetic split it prints; the >50% synthetic warning is expected.

**Record for the paper:** the 8C breakdown counts. "N trajectories, M malformed calls, of
which K were the terminal `submit_patch`" is a sentence §5.6 currently cannot make.

### A2. Master notebook

Order is **4 → 4B → 0F → 0C → 6**. 0F now precedes 0C: it derives the schemas and reloads
the module, so 0C's receipt describes what you will actually train on.

- [ ] **Cell 4** — re-paste from `src/chat_format.py` (stage1d), run.
- [ ] **Cell 4B** — re-paste, run. Must print `✓ chat_format.py is current (version
      2026-11-10.stage1d)` plus the schema source and the 9 tool names.
      It now also asserts `node_id` cannot reappear.
- [ ] **Cell 0F** — re-paste, run. The `real_tools.json` already on Kaggle was written by
      0F **v1**, which typed `list[str]` as `"string"`. chat_format now refuses any file
      not stamped `2026-11-10.0F-v2` and says so. Re-running 0F replaces it.
      Expect all nine rows `✓`, then `chat_format reloaded. Schemas now from:
      /kaggle/working/real_tools.json (generator 2026-11-10.0F-v2)`.
- [ ] **Cell 0C** — run (no re-paste; unchanged). The old receipt is invalid twice over —
      version and fingerprint both moved. Expect **`outcome=clean`** this time, because
      0F set `real_tools`. If it still says `inconclusive`, 0F did not reload; re-run 0F.
- [ ] **Cell 6** — re-paste. Carries the Gate 8 call. Do not run training today.
- [ ] **Cell 11** — re-paste if not done since the audit fix (`git apply --check` +
      RESOLVED columns).

**Stop here.** Training is Part D (post-paper). The point of today is that the next
training run starts from verified schemas and clean data.

---

## Part B — The paper (Nov 11, all day)

Most of §6, §7 and §8 of `PAPER_FINALIZATION.md` are already written into the manuscript;
the boxes were never ticked. What genuinely remains:

### B1. Fold today's F6 evidence into §5.6 (~30 min) — DO FIRST
- [ ] Add the 8C counts: how many trajectories, how many malformed calls, how many were
      the terminal `submit_patch`. This converts "every trajectory demonstrates a rejected
      call" from an inference about the generators into a measured property of the data.
- [ ] Add 0C's `outcome=clean` receipt as the verification artefact Gate 8 produces.
- [ ] Decide §7: may it state that F6 accounts for the score-0 submission pattern? With
      the enumeration complete and the data counts in hand, **yes** — but phrase it as
      *accounts for*, not *caused*, since no corrected run exists.

### B2. The three numbers still missing (~2–3 h)
- [ ] **2.2** Recompute historical submission rates under the strict metric
      (applies-cleanly AND tests-pass). This is the honest version of the 85.7% → 0.0%
      claim in §5.5 and it needs a table, not a sentence.
- [ ] **2.3** Per-task breakdown from the score-0 run: did `edit_file` succeed, did the
      patch apply, did tests run. This is now *more* valuable than it was yesterday —
      with F6 known, a per-task trace showing graph calls failing first is direct
      corroboration rather than speculation.
- [ ] **2.4** All runs' loss curves in one figure against resolution. **This is the
      paper's central image** and §7.2 currently describes it in prose.

### B3. Table 1 — the one positive result (~2 h)
- [ ] **5.1** Re-run with task-specific negatives.
- [ ] **5.2** Confirm the judge is the **base** model, never `main_lora`.
- [ ] **5.3** Report stratified by negative class.
- [ ] **5.4** State n, how negatives were generated, and that they are synthetic.
- [ ] If it does not re-run in time: mark Table 1 **provisional** in the caption and say
      so in §9. Do not ship it unmarked.

### B4. OPTIONAL — the F1 ablation
- [ ] Two adapters identical but for stop-token masking; report ceiling-hit rate and
      submission rate. **Cut this first if time runs short.** §9.2 already labels F1
      observational, which is defensible. A rushed ablation is not.

### B5. The honesty pass (~1 h) — NEVER CUT
- [ ] **9.1** Every number traces to a log or table. Anything that does not, delete.
- [ ] **9.2** Read start to finish as a hostile reviewer: "what measurement supports this?"
- [ ] **9.3** Reconcile `PROGRESS_LOG.md` and `competition_plan.md` — both still contain
      the old triumphal claims the manuscript now retracts. A reviewer who opens the repo
      will find them.
- [ ] **Fix the five-vs-six inconsistency.** §1 says "five post-training runs" and then
      lists six leaderboard scores. You know which is the baseline; I do not.
- [ ] Confirm §5.6's second-order admission survives editing — the `list[str]` mistyping
      in our own derivation script. It is the most credible paragraph in the section.

---

## Part C — Nov 12

- [ ] Artifact release: `src/chat_format.py`, `tests/test_chat_format_masking.py`,
      `tests/test_tool_schema_gate.py`, Cells 0C–0F, Cell 8C. The gates are the
      contribution; ship them runnable.
- [ ] Final read.
- [ ] Submit.

---

## Part D — after the paper (Dec 2 competition track)

Not for Nov 11. See `TRAINING_FIX_PLAN_DEC2.md`.

1. Stage 2: chunked masked loss, restore `MAX_LEN=8192`.
2. Stage 3: clean SFT on corrected data → submission 1.
3. Stage 4: grounded-only ablation → submission 2.
4. Upload `src/` as a Kaggle Dataset. Three inline copies have drifted so far
   (`chat_format.py`, `train_sft.py`, Cell 8A's trajectory builder); two have already
   cost a run.

---

## What today changed, in one paragraph

F6 — interface transcription drift — is confirmed and corrected. Five of nine tools
diverged: `get_code_subgraph` takes `nodes: list[str]` not `node_id`, `get_code_neighbors`
takes `node`/`edge_type` not `node_id`/`direction`, `search_similar_code` takes `k` not
`top_k`, and `submit_patch` takes **no arguments** while every trajectory supplied a
`description`. The last of these means the terminal action of every demonstrated episode
was malformed. Schemas are now derived from the harness rather than transcribed, the
derivation is version-gated, the trainer refuses to start without a clean receipt, and the
training data is rewritten to match. None of it has been validated by a run, which is
stated in §9.9 and must stay stated.
