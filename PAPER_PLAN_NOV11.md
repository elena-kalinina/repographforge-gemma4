# Paper Plan — Nov 11, 5–6 hours to a submittable manuscript

Target: `paper/manuscript.md`, Paper Track deadline Nov 12.
Companion: `RUNBOOK_NOV11.md` (cell locations), `PAPER_FINALIZATION.md` (history).

**Goal of the day.** No new experiments can land in six hours. The work is (a) promote
the general claim the data already supports so the paper stops reading as a bug list,
(b) add one quantitative table built from existing measurements, and (c) close three
evidentiary holes a hostile reviewer will find. Everything below is local CPU except two
clearly marked Kaggle tracks that run in parallel while you write.

---

## 0. The reframe in one paragraph (write this first, then make the paper match it)

> Agentic SFT failures hide below the **resolution floor of the training loss**. Each of
> the six defects we hit lives in a few percent of the supervised tokens (the stop token;
> a handful of parameter names; the format of a terminal call), so a loss that fell from
> 2.05 to 0.39 could not register them. In two of six cases the model learned the tokens
> *perfectly against a wrong target*, where loss is unhelpful by definition. The defects
> also **queue**: one masks the next, so repairing one exposes rather than fixes. And for
> one of them the ground truth lies **outside the pipeline** (the harness interface), so
> no internal consistency check can substitute for derivation. The remedy is a set of
> structural gates run before the GPU spend; applied retroactively they would have
> refused every run the loss blessed.

Three nouns to use consistently: *resolution floor*, *queueing*, *external ground truth*.

---

## 1. Timeline (start to finish, ~5.5 h local; Kaggle tracks run alongside)

| Clock | Local (critical path) | Kaggle, in parallel |
| :--- | :--- | :--- |
| 0:00–0:15 | §2 Regenerate and commit the loss figure | Start CPU session; run Cells 12A → 12B (§7) |
| 0:15–1:00 | §3 Honesty fixes: F2, V_exec, run count, 85.7% | *(optional)* start GPU session, vLLM base model, Data-prep Cells 4 → 5 (§8) |
| 1:00–3:30 | §4 Retroactive gate audit table | Check 12B output; fold into §5.6 / §7 |
| 3:30–4:30 | §5 Abstract + §1 rewrite around §0 | Check Table 1 re-run; fold or mark provisional |
| 4:30–5:00 | §6 Resolution-floor arithmetic paragraph | — |
| 5:00–5:30 | Final hostile read; commit; stop | — |

Hard rule: at 5:00 stop adding and start reading. Whatever is not in by then is cut.

---

## 2. Regenerate and commit the central figure (15 min)

The paper's §7.2 references `experiments/plots/loss_curves.png`. It is **not in the repo**
(`experiments/plots/*` is gitignored), so the artifact release currently ships without
its central figure.

- [x] Regenerated 06:11 Nov 11 with `.venv/bin/python` (3.12). Parser fixed
      (`src/parse_training_log.py` had a 3.12-only f-string). Output matched: 7 runs,
      73 points; finals 0.7195 / 1.057 / 1.414 / 0.6377 / 1.067 / 0.5711 / 0.349.
  ```bash
  .venv/bin/python src/parse_training_log.py experiments/logs/train_log_rft_sft.log \
      --csv experiments/logs/train_loss_points.csv \
      --plot experiments/plots/loss_curves.png
  ```
- [x] PNG inspected: two panels (per-run curves; final-loss bars with RFT in red, n=4
      annotated). Both RFT passes visibly short and high. No resolution axis — **add the
      caption line** "Resolution across these checkpoints: 0/14, 0/14, 0/14, 1/14, 1/14,
      0/14, 0/14" when placing the figure in §7.2.
- [x] `.gitignore` exceptions added for `loss_curves.png` and `train_loss_points.csv`.
- [ ] Commit: `git add experiments/plots/loss_curves.png experiments/logs/train_loss_points.csv`
      (no `-f` needed now).

---

## 3. Honesty fixes (45 min) — do these before any writing

### 3.1 F2's evidence is invalid as stated — rewrite §5.2

The §5.2 argument is: median trajectory 831 tokens < preamble 1,110 tokens, therefore the
preamble cannot have been present. But the 831 was measured in master Cell 0A on
`json.dumps(messages)`, not on rendered training sequences (Cell 0A line ~114:
`txt = ex.get("text") or json.dumps(ex.get("messages", []))`). A raw-message median of 831
is fully consistent with a present preamble (831 + 1,110 = 1,941 < 2,560 cap). The
drop-count argument (5 of 304 dropped) is inconclusive for the same reason.

What *is* established:
- Runs 1–5: declarations absent **by construction** — the original code never passed
  `tools=` at all (`DIAGNOSIS.md` C2). Solid.
- Runs 6–7: the `try/except Exception: pass` fail-open path existed. Whether it fired is
  **undetermined** from the records.
- Cell 0C (Nov 10): the corrected renderer emits 1,110 tokens with declarations present.

**DONE 06:20 Nov 11** (decision: accepted). Applied in:
- [x] §5.2 rewritten: mechanism paragraph now distinguishes runs 1–5 (no `tools=` at
      all) from 6–7 (fail-open renderer); new "What is established, and what is not"
      paragraph; the retracted median argument is named as wrong and the Cell 0A
      measurement identified as the source; "margin honestly" paragraph replaced by the
      double correction (1,830 estimate, then the invalid comparison); interaction-with-
      length-policy paragraph softened to "hazard by construction"; assertion updated to
      the live probes (`old_string`, `nodes`, `submit_patch` — the old one still said
      `node_id`, which is itself an F6 relic).
- [x] Abstract F2 clause; §5 table row; §4.4 bullet (states the lengths are raw-JSON);
      §5.6 "masked by F2" and §11 queueing sentence reworded to "while declarations were
      absent"; Appendix A closing paragraph replaced; **§9.11 added**.
- [x] `STATUS_REVIEW.md` §1(a), `paper/REFRAME.md` §4, `PAPER_FINALIZATION.md` 6.2,
      `RUNBOOK_NOV11.md` §6 — each carries a dated re-resolution with the superseded
      wording quoted, not deleted. `TRAINING_FIX_PLAN_DEC2.md` 1.1 never made the median
      claim; left alone.
- [x] `src/chat_format.py` comment block rewritten; version bumped to
      `2026-11-11.stage1f`; notebook Cell 4 mirror and Cell 4B `EXPECTED_VERSION` updated
      and verified byte-identical. Schema fingerprint unchanged (`649592b1b964a77b`).
      **Consequence:** the Kaggle receipt is invalidated by version; run 4 → 4B → 0F → 0C
      once before the next training session (already required; see RUNBOOK §5).
- [x] Count kept at six; §5.2 closes with the two-tier status sentence.
- [ ] Re-read §5.2 once in the final pass — it is now the longest subsection and may
      want one paragraph trimmed.

### 3.2 Table 1's "V_exec" was never executed — fix §3.3 / §6

`oracle_label` is assigned by construction in `src/candidate_patch_generator.py`
(gold → True, truncated/drift → False). No test ran. Truncated negatives (`lines[:len//2]`)
are frequently malformed diffs that would not apply.

- [ ] §6 **Setup**: replace "compared against $V_{\text{exec}}$" with "compared against
      labels assigned by construction: the gold maintainer fix is labelled PASS because
      it is the patch the official `test_patch` was written against; truncated and drift
      patches are labelled FAIL without execution, and the truncated class is frequently
      not a syntactically valid diff."
- [ ] §6 **Interpretation**: drop "agreement with execution"; say "agreement with the
      constructed labels." Keep the direction-of-effect claim; drop the "substantial
      agreement (>0.70)" psychometric framing.
- [ ] §6 **Threats**: add threat 4 — *Oracle by construction, not execution.* And threat
      5 — *Rubric leakage*: rubrics were generated from the problem statement **and the
      gold patch** (`src/rubric_generator.py` prompt), so $V_{\text{graph}}$ sees
      information derived from the answer. Legitimate for RFT filtering (gold is
      available at training time); not a fair judge of unseen patches. Say both.
- [ ] §3.3: keep the $V_{\text{exec}}$ definition but add "not run in this paper; see §6."
- [ ] README "one surviving positive result" block: same wording change.

### 3.3 Run count and the 85.7% pairing

- [ ] §1 line ~24: "Over five post-training runs" → "Over seven training runs (five SFT,
      two RFT passes)". The six leaderboard scores are submissions, not runs; say so.
- [ ] Abstract and §5 table: "reporting 85.7% where the true resolution rate was 0.0%" is
      a mismatch. 85.7% in §7.1 is *target-file localization* for run 5, whose resolution
      was 7.1%; the 71.4%/85.7% *submission* figures came from the `args == {}` audit on
      an earlier run. Pick one true pair. Safest: "reporting submission rates of 71–86%
      on runs whose resolution rates were 0–7%". Apply in abstract, §5 table, §5.5.

---

## 4. Retroactive gate audit table (2.5 h) — the new result

Build a **runs × gates** matrix: for each of the seven training runs, would each of the
nine gates (§8) have refused it *before the model loaded*? This converts the taxonomy
into a checkable claim and is the paper's second figure.

### 4.1 Evidence you already have (fill these cells first)

| Run (from §7.2) | Gate 1–2 (coverage, stop token) | Gate 3 (leakage) | Gate 4 (train/eval identity) | Gate 5 (survival / steps) | Gate 6 (inter-stage format) | Gate 7 (falsifiable eval) | Gate 8 (schemas vs harness) | Gate 9 (data vs schemas) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 First SFT, Oct 30 | REFUSE — masking loop (DIAGNOSIS B1) | REFUSE — whole sequence supervised | REFUSE — no `tools=`, no system prompt (C2) | pass (112, 42 steps) | n/a | REFUSE — audit counted attempts | REFUSE — `node_id` etc. | REFUSE — `submit_patch(description)` in all data |
| 2 SFT 2nd, Oct 31 | REFUSE (same loop) | REFUSE | REFUSE (C2) | pass | n/a | REFUSE | REFUSE | REFUSE |
| 3 RFT, sub 1 | REFUSE — no masking at all (B3) | REFUSE | REFUSE | **REFUSE — 4 examples, 4 steps** | **REFUSE — 0 tool calls in accepted set** | REFUSE | REFUSE | REFUSE |
| 4 SFT scaled, Nov 4 | REFUSE | REFUSE | REFUSE | pass | n/a | REFUSE | REFUSE | REFUSE |
| 5 RFT, Nov 4 | REFUSE | REFUSE | REFUSE | **REFUSE** | **REFUSE** | REFUSE | REFUSE | REFUSE |
| 6 SFT corrected tool calls, Nov 5 | *inferred pass* (chat_format v1) | inferred pass | **unknown** (fail-open path) | pass (299 kept) | n/a | REFUSE | REFUSE | REFUSE |
| 7 SFT, sub 3, Nov 9 | inferred pass | inferred pass | unknown | pass | n/a | partial (Cell 11 strict count, no resolved) | REFUSE | REFUSE |

Mark every cell **established** (quote the file/line or log line), **inferred** (say from
what), or **unknown**. Do not fill an unknown with a guess — that is F5.

### 4.2 Steps

- [ ] Confirm which `chat_format` version runs 6 and 7 used. Evidence: git log dates vs
      log dates (`git log --since=2026-11-04 --until=2026-11-09 -- src/chat_format.py`).
      If the masking fix (commit `34c127e`) predates run 6, Gates 1–3 are "inferred pass"
      for 6–7; otherwise REFUSE.
- [ ] Gate 9 for every run: the Cell 8C sweep found `submit_patch` with seven argument
      names and `node_id` in every grounded and synthetic trajectory. Since all seven runs
      trained on files from that corpus, Gate 9 is REFUSE for all seven — **established**.
- [ ] Gate 8 for every run: F6 was in the declarations until Nov 10 — REFUSE for all
      seven, **established** (Cell 0D/0E output in `TRAINING_FIX_PLAN_DEC2.md` 1.1).
- [ ] Gate 7: no run had tests-pass in its local eval — REFUSE for all seven.
- [ ] Write the table into a new §7.4 "Retroactive gate audit" with one paragraph:
      *N of 7 runs would have been refused by at least one gate with established
      evidence; the loss curve refused 0 of 7.* Count N honestly (it is 7).
- [ ] Add one sentence to the abstract and one to §11.
- [ ] Optional (30 min): generate the table as a small heat-map PNG with matplotlib,
      three colours (refuse / pass / unknown). Commit with `git add -f`.

---

## 5. Abstract and §1 rewrite (1 h)

The current abstract is a six-item list. Replace with the §0 paragraph structure:

1. One sentence of setting (Gemma 4 31B, repository bug-fixing, seven runs, loss
   converged, resolution ≈ 0).
2. The resolution-floor claim, with the 1–4% figure from §6 below.
3. The queueing claim (F6 hidden by F2).
4. The external-ground-truth claim (harness interface; derive, don't transcribe).
5. The retroactive audit result (7 of 7 refused by gates, 0 of 7 by loss).
6. One sentence on the verifier study, stated with its by-construction caveat.
7. "We release the gates as runnable assertions."

Move the F1–F6 enumeration out of the abstract into §5's lead table (it is already there).

- [ ] Rewrite abstract.
- [ ] Rewrite §1 paragraphs 2–3 to carry the three nouns from §0.
- [ ] Retitle if you want: *"Below the Resolution Floor: Why Agentic SFT Fails While the
      Loss Converges"* — or keep the current title. Either is fine; the abstract matters more.
- [ ] §1.2 Contributions: lead with the resolution-floor analysis and the retroactive
      audit; the taxonomy becomes item 3, the gates item 4, the verifier study item 5.

---

## 6. Resolution-floor arithmetic (30 min) — new §5.0 or opening of §8

Use only numbers already measured:

- Grounded trajectory structure (`src/build_real_sft.py`): 5 assistant turns for a
  single-hunk task, 7 for two hunks → 5–7 stop tokens.
- Supervised fraction gate: 3–45% of tokens; test replay shows 38.5% on the measured
  template (`tests/test_chat_format_masking.py` output). Take a typical supervised span
  of 250–450 tokens for a median trajectory.
- F1: 5–7 stop tokens / 250–450 supervised tokens ≈ **1–3% of the loss mass**.
- F6: `node_id` + `description` + `direction` ≈ 2–4 tokens per call × 5 calls ≈ 10–20
  tokens ≈ **2–5%**, and these tokens were learned *correctly* against a wrong target.
- F3: the RFT stage's loss *rose* (1.414 vs 0.720; 1.067 vs 0.638) — the one case the
  loss did signal, and nobody compared across stages.

- [ ] Write the paragraph with these ranges; state they are ranges from measured
      structure, not from a per-token loss decomposition (which you did not run).
- [ ] Add the F3 observation as the exception that proves the rule: the loss *did* carry
      information once, across stages, and the pipeline had no comparison step.

---

## 7. Kaggle CPU track — trace forensics (45 min, run while writing)

Only empirical addition worth doing today. Turns F6 from "diagnosed" into "corroborated."

- [ ] CPU session. Run master **Cell 12A** (lines 2448–2483). Paste the schema.
- [ ] If the trace shape differs from what 12B probes for (`name`/`function.name`,
      `arguments`), patch 12B's extraction before trusting it.
- [ ] Run **Cell 12B** (lines 2495–2577). The lines that matter:
      `submitted with NO edit : n/14` and the first error observation per task.
- [ ] If the first graph call errors with an unexpected-keyword / validation message on
      most tasks, add a short "Trace corroboration" paragraph to §5.6 with the count and
      one verbatim error string, and let §7 say F6 **accounts for** the score-0 pattern.
- [ ] If 12B cannot parse some traces, report the parsed denominator, not 14.
- [ ] Also from the traces: capture **one real observation string** per tool
      (`read_file`, `get_code_subgraph`, `edit_file`, `submit_patch`). Save to
      `experiments/logs/real_observations.json`. Not for the paper — this is the first
      input to `EXPERIMENT_PLAN_DEC2.md` Stage 1 and it costs nothing now.

---

## 8. Kaggle GPU track — Table 1 re-run (OPTIONAL, ~2 h wall, low attention)

**Decision Nov 11: deprioritised, as recommended.** Run it only if the §7 CPU session is
finished and the GPU session costs no writing attention. The honest Table 1 is the one
`EXPERIMENT_PLAN_DEC2.md` Stage 4 produces on real rollouts with real execution labels.

Fixes judge contamination and the constant drift string. Does **not** fix the
by-construction oracle (§3.2) — say so regardless of outcome.

- [ ] GPU session. Ryan's Section 1–3, then Section 4 serving the **base** model (no
      adapter). Confirm `/v1/models` lists only the base id.
- [ ] Data-prep **Cell 4** (task-specific negatives), then **Cell 5** (base judge,
      stratified). ~386 candidates × 2 judges at 4 workers — budget 30–60 min after
      server start.
- [ ] Copy the resulting `verifier_alignment_table1.json` into `experiments/logs/`
      alongside the pilot file (rename pilot to `..._pilot.json`; keep both).
- [ ] If it finishes by 4:30: replace the §6 table with the new numbers, add the
      stratified FP rows, state judge id and n. **If the gain shrinks, the smaller number
      is the result.**
- [ ] If it does not finish: leave the pilot table, keep "provisional", add one sentence
      that a base-judge re-run was started and will be reported in the Dec 2 artifact.
- [ ] Do not let this track pull attention from §3–§5. It is the least important item here.

---

## 9. Final hostile read (30 min) — checklist

- [ ] Every number traces to `experiments/logs/` or a quoted file/line.
- [ ] No sentence says "V_exec" where the label was assigned by construction.
- [ ] No sentence says the median trajectory was shorter than the preamble.
- [ ] "Seven runs", "six submissions", consistent everywhere.
- [ ] The 85.7% appears only with its correct partner.
- [ ] §5.6 second-order admission (`list[str]` mistyping) still present.
- [ ] §9 Limitations gained: oracle-by-construction; rubric leakage; F2 realization
      undetermined; retroactive audit cells marked inferred/unknown where applicable.
- [ ] Loss figure present in repo; gate-audit table present.
- [ ] Commit with a message naming the three honesty fixes explicitly.

---

## 10. What is deliberately NOT in this plan

- F1 controlled ablation (needs two training runs).
- Re-running the agent with corrected schemas (that is `EXPERIMENT_PLAN_DEC2.md`).
- Any end-task claim. The Dec 2 result goes in the Dec 2 artifact and the presentation,
  not in tomorrow's paper.
