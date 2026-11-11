# Paper Finalization Checklist — Nov 10 → Nov 12

Framing: `paper/REFRAME.md`. Target: Kaggle Paper Track, **Thursday Nov 12**.
Everything here is CPU except §3, which is optional.

**Rule for the next 48 hours:** no claim goes in that isn't backed by a number you can
point at in a log or a table. The reframe's whole value is that it's checkable.

---

## Day 1 — Nov 10: settle the facts, lock the structure

### 1. Resolve the open question that changes F2's wording
- [x] **1.1** Paste the Phase 0.1/0.2 outputs into `experiments/logs/phase0_facts.md`
      (you ran them; they just aren't written down). Specifically: the rendered template
      string, the real turn-open/turn-close token spellings, and the ADK tool schema dump.
- [x] **1.2** Settle whether tool schemas are actually in the training text:
      ```python
      from chat_format import render, COMPETITION_TOOLS
      r = render(msgs, tokenizer, tools=COMPETITION_TOOLS)
      print(len(tokenizer(r, add_special_tokens=False)["input_ids"]))
      print("read_file" in r, "old_string" in r, "node_id" in r)   # tools present?
      ```
      If those are `False`, reading (a) in `REFRAME.md` §4 holds — the `try/except` ate
      the tools and C2 was never really applied.
- [ ] **1.3** Diff the real ADK tool schemas (from 1.1) against `COMPETITION_TOOLS`
      field by field. Record every mismatch in `phase0_facts.md`. **This is also the
      top item in the Dec 2 plan** — if names differ, it likely explains the score-0 run.
- [x] **1.4** Record the training-set composition: `sft_combined.jsonl` = 115 grounded +
      ~184 synthetic screenplays = 299. State the **62% synthetic** share; it belongs in
      Limitations and it matters for interpreting every run.

### 2. Numbers the paper needs
- [ ] **2.1** Token-length distribution of trajectories with and without the preamble
      (p50/p90/p95/p99). Quantifies F2 and the old 1536 truncation.
- [ ] **2.2** Recompute historical submission rates with the strict metric. Produce the
      pair **(legacy count, strict resolve rate)** for every checkpoint in Table 2. This
      is the F5 evidence.
- [~] **2.3** Traces LOCATED on Kaggle (2026-11-10): `traces/trace_<task>.json` for all 14
      holdout tasks, plus `<task>.log`. Cells 12A (schema discovery) and 12B (per-task
      table) added to the master notebook. 12B reports the F6 signature directly:
      *submitted with no successful edit*. Run 12A first and paste the schema — the
      extraction probes generically and must be checked before its numbers are trusted.
- [~] **2.4** `train_log_sft_rft.log` RECOVERED from a Kaggle notebook version
      (2026-11-10). `src/parse_training_log.py` added — tolerant of HF `Trainer` dict
      lines, JSONL, and `loss=`/`step=` key-value forms, and it says which pattern matched
      rather than returning an empty series. Writes a CSV and the §7.2 figure:
      ```bash
      python3 src/parse_training_log.py experiments/logs/train_log_sft_rft.log \
          --plot experiments/plots/loss_curve.png
      ```
      - [ ] Download the log into `experiments/logs/` — needed for 9.1 (every number
            traces to a log) and 9.4 (artifact release) regardless of the figure.
      - [x] **Provenance: resolved as far as the records allow (2026-11-10).** The log
            holds the whole campaign — 7 dated runs, 73 loss points. Three name
            themselves as submissions 1–3; the other four cannot be attributed. Decision:
            **leave unmapped.** §7.2 states the gap explicitly and §9.9 records it as a
            limitation, rather than inferring the mapping from chronology.
      - [x] Stage boundaries marked: the figure colours the two RFT passes red. Both
            trained on 4 examples for ~28 s and finished at ~2× the loss of the SFT they
            overwrote (1.414 vs 0.720; 1.067 vs 0.638). Folded into §5.3.
      - [x] **2.4 COMPLETE.** `experiments/plots/loss_curves.png`,
            `experiments/logs/train_loss_points.csv` and the source log are committed.
- [ ] **2.3 (original wording)** From the score-0 run's traces: per task, did `edit_file` succeed, was the
      submitted diff non-empty, did it apply. Even without test execution this
      distinguishes "never edited" from "edited wrongly" — and it's the F4/F5 evidence.
- [ ] **2.4** Pull the loss curves for every run into one figure. **This is the paper's
      money figure:** every curve converges, every Pass@1 is ~0. One plot, two axes.

### 3. OPTIONAL — the F1 ablation (only if quota and time allow)
- [ ] **3.1** Two adapters, identical but for stop-token masking. Small is fine: a few
      dozen trajectories, 1 epoch.
- [ ] **3.2** Evaluate both on the same tasks. Report **% responses hitting
      `max_output_tokens`** and **strict `submit_patch` rate**.
- [ ] **3.3** If it doesn't run: say so. Report F1 observationally and label it
      "not a controlled ablation; primary planned experiment". Do not imply otherwise.

### 4. Structure
- [x] **4.1** Restructure `paper/manuscript.md` to the `REFRAME.md` §6 outline.
- [x] **4.2** Draft §7 (structural gates) — now §8 first — it's the takeaway and the hardest to
      write well. Everything else is reporting.

---

## Day 2 — Nov 11: write, re-run Table 1, purge

### 5. Table 1 — the surviving positive result
- [ ] **5.1** Re-run with task-specific negatives (`src/candidate_patch_generator.py` 7.2,
      already implemented). The old constant `fix.py` string made a third of the suite
      trivially separable.
- [ ] **5.2** Confirm the judge is the **base** model, never `main_lora` (7.3).
- [ ] **5.3** Report **stratified by negative class** (`partial_contract_omission` vs
      `sycophantic_drift`), not just pooled. Pooled numbers hide which class drives it.
- [ ] **5.4** State n, the generation procedure for negatives, and that negatives are
      synthetic. If the gain shrinks with honest negatives, **report the smaller number** —
      a modest well-measured gain survives scrutiny; an inflated one doesn't.

### 6. Write the taxonomy
- [ ] **6.1** §F1 stop-token starvation — include the six-line masking loop verbatim; the
      bug is visible to any reader, which is what makes it convincing.
- [x] **6.2** §F2 preamble–budget — **re-settled 2026-11-11.** The 2026-11-10 wording
      ("median 831 < preamble 1,110, so the preamble was absent") was wrong: the median
      was measured on raw message JSON, preamble excluded by construction. Now stated as
      realised by construction in runs 1–5 (no `tools=` at all) and undetermined in runs
      6–7 (fail-open path, uninstrumented). Retraction stated in §5.2, §4.4, §9.11,
      Appendix A, STATUS_REVIEW and REFRAME rather than quietly swapped.
- [ ] **6.3** §F3 inter-stage format drift — include the markdown-fence regex verbatim.
- [ ] **6.4** §F4 serialization fragmentation — the four-format table from `DIAGNOSIS.md` §C1.
- [ ] **6.5** §F5 unfalsifiable metrics — the `or args == {}` line and the 85.7% → 0.0% gap.
- [ ] **6.6** Each subsection ends with its **corrective assertion** as runnable code.

### 7. Purge unsupported claims (`REFRAME.md` §5)
- [ ] **7.1** Remove "verified" / "pristine" for trajectories. Filter was `len(msgs) >= 4`.
- [ ] **7.2** Remove the 2× `V_graph` gain unless 5.2 re-ran it.
- [ ] **7.3** Relabel `requests_7505` / `6644` / `7433` as qualitative, n = 1.
- [ ] **7.4** Move Table 3 / multi-scale claims to Future Work. Delete any unrun number.
- [ ] **7.5** Confine 71.4% / 85.7% to §F5, as the error being reported.
- [ ] **7.6** Grep for superlatives: "breakthrough", "massive", "perfectly", "pristine",
      "dramatically". The reframe's credibility comes from flat, exact prose.

### 8. Limitations — write it properly, it's load-bearing
- [ ] **8.1** Pass@1 ≈ 0 means **no claim** about whether graph anchoring helps end-task
      performance. Say it in those words.
- [ ] **8.2** Single model family, single task family, two source repos.
- [ ] **8.3** 62% of SFT data was unverified synthetic trajectories (1.4).
- [ ] **8.4** F1 observational unless §3 ran.
- [ ] **8.5** Table 1 negatives are synthetic, not harvested from real failed rollouts.

---

## Day 3 — Nov 12: verify and submit

- [ ] **9.1** Every number in the paper traces to a log or table in `experiments/logs/`.
- [ ] **9.2** Read the manuscript start to finish as a skeptical reviewer. Ask of each
      claim: *what would I have to run to falsify this, and did they run it?*
- [x] **9.3** Reconcile `PROGRESS_LOG.md` and `competition_plan.md` with the final text.
      **Done 2026-11-10.** Both carry a superseded banner naming the four things the
      manuscript establishes that they predate (resolution ~0; nothing execution-verified;
      F1–F6 active; no end-task claim), plus inline `[RETRACTED]` notes on the specific
      claims: "pristine"/"verified" trajectories, "perfectly adheres" on `requests_7505`,
      and the verified-trajectory framing in the method summary. Deliberately **not**
      rewritten otherwise — a log edited silently after the fact has no evidentiary value,
      and the visible retractions are themselves consistent with the paper's argument.
      they currently contain the old triumphal claims and are in the repo you're releasing.
- [ ] **9.4** Artifact release: `src/chat_format.py`, the assertion harness, the diagnostic
      notebook cells. The harness **is** the contribution — make it easy to find and run.
- [ ] **9.5** Submit.

---

## Scope discipline

Do **not** attempt before Nov 12: a working agent, a leaderboard-worthy checkpoint, Table
3, or Phase 5/6 grounded rollouts. Those are the Dec 2 plan. The deadline on Thursday is a
*writing* deadline, and the single biggest risk to it is spending Wednesday on a GPU run
instead of on §6 and §8.


---

## Added 2026-11-10 — failure mode F6

- [x] **F6 written into the manuscript.** Interface transcription drift: the declared tool
      parameters are a hand-written copy of the harness interface and diverged from it.
      Evidence quoted verbatim from `swegemma/tools/graph.py`
      (`node` / `edge_type`, against our `node_id` / `direction`).
      - [x] §5.6 new section; label-direction inversion renumbered to §5.7
      - [x] §5 taxonomy table row
      - [x] Abstract, §1, §1.1, §1.2 updated from five defects to six
      - [x] §8 Gate 8 — interface identity against the harness, with the receipt design
      - [x] §9.9 — F6 is diagnosed, not repaired; no re-run has been done
      - [x] §11 conclusion, including the "silent failures queue" lesson
- [x] **Full nine-tool enumeration folded into §5.6** as a table. 5 of 9 diverge at
      parameter level. `submit_patch` takes no arguments while every trajectory supplies
      `description`; `get_code_subgraph` takes `nodes: list[str]`, so the dataset fix is
      a value transformation, not a rename. Also records the second-order failure: the
      derivation script typed `list[str]` as `"string"` by substring match.
- [x] **§5.4 counts filled from the Cell 8C v3 run (2026-11-10).** 1,933 tool calls,
      503 (26.0%) string-encoded. Grounded trajectories discarded 0 of 114; the two
      model-generated files discarded 27.7% and 30.5% as unrepairable. §4.2 updated to
      match. **No TODOs remain in the manuscript.**
- [ ] Decide whether §7 should state that F6 accounts for the score-0 submission
      pattern. The enumeration is now complete and the data counts are in, so this is
      answerable — hold only until item 2.3 (per-task traces) is attempted.
- [ ] **Inconsistency noticed, not yet resolved:** §1 says "five post-training runs" but
      lists six leaderboard scores (0.00, 0.00, 0.00, 0.03, 0.01, 0.00). Reconcile.
