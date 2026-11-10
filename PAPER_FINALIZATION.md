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
- [ ] **2.3** From the score-0 run's traces: per task, did `edit_file` succeed, was the
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
- [ ] **6.2** §F2 preamble–budget — wording per 1.2.
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
- [ ] **9.3** Reconcile `PROGRESS_LOG.md` and `competition_plan.md` with the final text —
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
