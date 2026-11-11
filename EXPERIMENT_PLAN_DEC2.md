# Experiment Plan — Nov 13 → Dec 2: Verifier Selection on Real Rollouts

**Supersedes** Stages 3–5 of `TRAINING_FIX_PLAN_DEC2.md`. Stages 1–2 of that document
(verification gaps, memory fix) remain prerequisites for the optional SFT arm (Stage 6
here) and are referenced, not repeated.

## The experiment

The original research question was: *do graph-anchored rubrics ($V_{\text{graph}}$) agree
with execution ($V_{\text{exec}}$) better than an unanchored judge ($V_{\text{free}}$), and
does that make them a better filter?* The original design answered it with RFT (three
training arms). That needs ≥50 accepted trajectories per arm from a working policy, which
is out of reach.

The same question answered without training: **Best-of-N at inference.**

1. For each held-out task, sample $N$ rollouts from a policy in the real sandbox.
2. Run the task's tests on **every** rollout → real $V_{\text{exec}}$ labels on real
   agent patches (not synthetic negatives, not labels by construction).
3. Score every rollout's patch with $V_{\text{free}}$ and $V_{\text{graph}}$ using the
   base model as judge.
4. Report (a) per-rollout agreement with execution (P/R/F1/κ, the honest Table 1), and
   (b) **selection accuracy**: if the verifier picks one rollout per task, how often is
   it a passing one — compared to pass@1, oracle pass@N, and random choice.

This is the original RQ1 with the two flaws of the pilot removed (synthetic negatives,
contaminated judge) and the one flaw neither manuscript admits removed (oracle by
construction). It needs a working eval loop and a policy that passes *sometimes*. The
base model with corrected schemas may be enough; an SFT arm is a bonus.

### Outputs

- `experiments/logs/rollouts/<policy>/<seed>/` — traces, predictions, verdicts.
- `experiments/logs/verifier_on_rollouts.json` — per-rollout judge scores + exec label.
- Two tables: verifier agreement on real rollouts; selection accuracy vs pass@k.
- One figure: pass@1 / V_free-selected / V_graph-selected / oracle pass@N, per policy.
- All of the above feed Act 3 of `PRESENTATION_PLAN.md` directly.

### Budget

| Stage | Where | GPU | Calendar |
| :--- | :--- | :---: | :--- |
| 0 Decisions and housekeeping | local | 0 | Nov 13 |
| 1 Harness discovery, real observations | Kaggle CPU | 0 | Nov 13–14 |
| 2 Eval wiring + zero-shot floor | Kaggle GPU | ~3 h | Nov 15 |
| 3 Best-of-N sampling, base policy | Kaggle GPU | ~3 h | Nov 16–17 |
| 4 Verifier scoring | Kaggle GPU | ~1.5 h | Nov 17–18 |
| 5 Analysis and tables | local | 0 | Nov 18–19 |
| 6 Optional SFT arm (repeat 2–4) | Kaggle GPU | ~6 h | Nov 20–26 |
| 7 Packaging, submission, export for demo | local + Kaggle | ~1 h | Nov 27–Dec 1 |

~8 GPU hours required, ~14 with the SFT arm. Every GPU session is preceded by a CPU
check that would have caught the previous failure (standing rule from the old plan).

---

## Stage 0 — Decisions and housekeeping (local, Nov 13)

- [ ] **0.1** Resolve the dual `agent.yaml`. `submission/agent.yaml` (schema: `name /
      model / adapter / instruction / tools`) is what master Cell 8 writes and what the
      grader reads. Delete nothing yet; confirm in Stage 1 which schema the ADK validator
      accepts, then delete the dead one.
- [ ] **0.2** `sub_agents/code_analyzer.yaml` is referenced by `agent.yaml` but is not in
      the repo. Find it in Ryan's starter bundle or remove the `agent_tool` entry. A
      dangling reference is a silent-failure candidate of exactly the F6 kind.
- [ ] **0.3** Decide the SFT data file (only matters for Stage 6): re-run data-prep Cell 9
      to rebuild `sft_combined.jsonl` from the cleaned sources (114 grounded + 82
      synthetic = 196, majority grounded). Record the choice in `experiments/logs/`.
- [ ] **0.4** Write `src/verifier_on_rollouts.py` skeleton now (CPU, no model): loads
      rollout directories, extracts final patch per rollout, joins exec verdicts, calls
      the two judges via the vLLM endpoint, writes JSON. Reuse `judge_v_free` /
      `judge_v_graph` from `src/eval_verifier_alignment.py` and `compute_alignment_metrics`
      from `src/rubric_verifier.py`. Test it offline against two hand-made fake rollouts.
- [ ] **0.5** Write `src/selection_analysis.py` (CPU): reads the JSON above, computes
      pass@1, pass@N (oracle), random-pick expectation, V_free-pick, V_graph-pick, with
      tie-break = shortest trajectory. Bootstrap CIs over tasks (14 tasks; CIs will be
      wide — report them anyway).
- [ ] **0.6** Rubric arms. The existing `evidence_rubrics.jsonl` was generated from problem
      statement **and gold patch** (`src/rubric_generator.py`). That leaks answer
      information into $V_{\text{graph}}$. Prepare a second rubric file generated from
      problem statement + graph neighbourhood **only** (no gold patch) — add a
      `--no-gold` flag to `rubric_generator.py` that omits the patch and instead includes
      the `get_code_subgraph` output for the symbols named in the issue. Generation runs
      in Stage 4 while the server is up. Report both arms; the no-gold arm is the fair one.

---

## Stage 1 — Harness discovery (Kaggle CPU, Nov 13–14)

Nothing in this stage needs a GPU, and everything in Stages 2–4 depends on it.

- [ ] **1.1** Run master Cells 12A/12B on the score-0 traces if not already done for
      the paper. Record the trace schema in `experiments/logs/trace_schema.md`:
      where tool calls live, where observations live, where the final patch lives.
- [ ] **1.2** Capture **one real observation string per tool** from the traces
      (`read_file`, `get_code_subgraph`, `get_code_neighbors`, `search_similar_code`,
      `edit_file`, `run_command`, `submit_patch`, `get_status`). Save to
      `experiments/logs/real_observations.json`. These replace the invented formats in
      `build_real_sft.py` if Stage 6 runs, and they tell you what the judge will see.
- [ ] **1.3** Find where Section 5 reports the **resolution verdict**. Read the
      `adk_eval_core` source in the wheelhouse (`inspect.getsource` on the evaluation
      entrypoint; grep for `resolved`, `passed`, `exit_code`, `FAIL_TO_PASS`). Write the
      exact key path into Cell 11's `RESOLVED_KEYS` / `_extract_resolved`. Do not probe
      at runtime again; know the field.
- [ ] **1.4** Find where Section 5 keeps the **repo checkouts** during verification (for
      `git apply --check`) and whether they persist after the run. If they do not, Cell 11
      must run `git apply --check` inside the verification step or skip the column
      honestly.
- [ ] **1.5** Confirm how Section 5 sets **sampling temperature** and whether it accepts a
      **seed**. Best-of-N needs $T>0$ and distinct samples; if the harness reads
      `configs/sampling.yaml`, set `temperature: 0.7, top_p: 0.95` there. If there is no
      seed control, distinct runs are distinct samples anyway — record wall-clock as the
      run id.
- [ ] **1.6** Confirm the harness's **tool-call budget** (`limits.max_tool_calls`: the
      progress log says 10, the prompt says 15). Set `max_turns` and the prompt's budget
      line to the real number.
- [ ] **1.7** Gate 4 for real: capture the **exact prompt** the harness sends on turn 1
      (system + tools + user) from a trace or by instrumenting the ADK call. Save as
      `experiments/logs/captured_eval_prompt.txt`. Byte-compare its first 300 chars
      against `chat_format.render([...], tokenizer, tools=COMPETITION_TOOLS)` in Stage 2.
      This is the one gate that has only ever been proposed.
- [ ] **1.8** Validator: run the ADK `submission.zip` validator on the current
      `sample_submission/` with no adapter and with a stub adapter dir. Learn which
      `agent.yaml` schema it accepts (closes 0.1).

**Exit criterion:** you can name the field that says PASS/FAIL, the directory that holds
the checkouts, the temperature knob, and the budget, each with a file:line citation.

---

## Stage 2 — Eval wiring and the zero-shot floor (Kaggle GPU session 1, ~3 h)

Purpose: prove the eval loop reports tests-pass end to end, and measure the floor every
later policy must beat.

- [ ] **2.1** CPU pre-check before the server starts: `tests/test_chat_format_masking.py`,
      `tests/test_tool_schema_gate.py`, master Cell 4B (version gate), Cell 0F → 0C
      (receipt must be `clean`). If anything refuses, stop; no GPU today.
- [ ] **2.2** Section 1–3 (Ryan), Cell 8 with **no adapter** (base model only; set
      `adapter:` to none or point at an empty identity adapter if the schema requires
      one — Stage 1.8 told you which). `EVAL_ARM = "C"` prompt. `max_output_tokens: 4096`.
      `temperature: 0.1` for this stage.
- [ ] **2.3** Section 4 (vLLM). Confirm `/v1/models` lists the base id only.
- [ ] **2.4** Gate 4 byte-compare (Stage 1.7). If the first 300 chars differ, fix
      `chat_format.render` or the prompt file before running anything else, and record
      the diff.
- [ ] **2.5** Run Section 5 on **one** task first (`requests_6644`, the one that has
      resolved before). Confirm Cell 11 prints `RESOLVED: ✓ PASS` or `✗ FAIL` — not
      `? unknown`. If unknown, go back to Stage 1.3; do not run the other 13.
- [ ] **2.6** Run all 14. Save traces, predictions, and verdicts to
      `experiments/logs/rollouts/base/greedy/`. Cell 11 output → `audit_base_greedy.md`.
- [ ] **2.7** Record: resolved count, applies-cleanly count, strict submit count, mean
      tool calls, ceiling hits, first-tool distribution. This is **Table 2 row "base,
      corrected schemas"** — a number the campaign never measured.

**Decision gate.**
- Resolved ≥ 1 → Stage 3 as planned.
- Resolved = 0 but applies-cleanly ≥ 3 → Stage 3 still worthwhile (verifier agreement
  on applying-but-failing patches is informative; selection accuracy will be degenerate
  and you say so).
- Applies-cleanly = 0 → the patch path is broken (edit_file arguments, diff extraction).
  Read three traces end to end. Fix. Re-run 2.5. Do not proceed to sampling.

---

## Stage 3 — Best-of-N sampling, base policy (Kaggle GPU session 2, ~3 h)

- [ ] **3.1** Same pre-checks as 2.1. Same Cell 8, but `temperature: 0.7, top_p: 0.95`.
- [ ] **3.2** Run Section 5 **four times** over the 14 tasks, each into its own directory:
      `experiments/logs/rollouts/base/sample_{1..4}/`. Each sweep ≈ 14 × (≤300 s agent
      + ≤300 s verification) worst case; realistic 30–45 min. If the session clock is
      tight, three samples is acceptable; two is not (selection becomes a coin flip).
- [ ] **3.3** After each sweep, run Cell 11 and save the audit. Check the verdict column
      is populated **every time** — a sweep with `? unknown` is discarded, not counted.
- [ ] **3.4** For each rollout, persist: task id, final patch text, exec verdict,
      applies-cleanly, tool-call count, total tokens, turn count, whether `edit_file`
      succeeded, first tool. One JSONL line per rollout:
      `experiments/logs/rollouts/base/rollouts.jsonl`.
- [ ] **3.5** Sanity numbers before leaving the session: pass@1 (mean over samples),
      pass@4 (any sample), number of tasks with mixed outcomes (some pass, some fail).
      **Mixed-outcome tasks are the only ones where selection matters.** If there are
      fewer than 3, the selection result will be anecdotal; the per-rollout agreement
      result (Stage 4) is still valid.

---

## Stage 4 — Verifier scoring (Kaggle GPU session 3, ~1.5 h)

- [ ] **4.1** Server up with the **base** model as judge. Confirm id.
- [ ] **4.2** Generate the **no-gold rubric arm** (Stage 0.6) for the 14 holdout tasks:
      `rubric_generator.py --no-gold --tasks-file val_ood_tasks.jsonl`. Save as
      `evidence_rubrics_nogold.jsonl`. Keep the existing gold-derived file as the second arm.
- [ ] **4.3** Run `src/verifier_on_rollouts.py` over `rollouts.jsonl` (56 + 14 greedy
      rollouts) with: $V_{\text{free}}$; $V_{\text{graph}}$ (gold rubric);
      $V_{\text{graph}}$ (no-gold rubric). Temperature 0, 512 max tokens, parse the JSON
      verdict strictly — a non-parseable response is recorded as `null`, not as FAIL.
- [ ] **4.4** Also score the **gold maintainer patch** for each of the 14 tasks under all
      three judges (14 more calls). This gives recall on known-good patches in the same
      run, comparable to the pilot Table 1 but on the holdout tasks.
- [ ] **4.5** Save `experiments/logs/verifier_on_rollouts.json` with judge id, prompt
      hashes, rubric arm, and timestamp. Download it. This file is the result.

---

## Stage 5 — Analysis (local, Nov 18–19)

- [ ] **5.1** `selection_analysis.py` → Table A, per policy and per rubric arm:

      | Judge | n rollouts | P | R | F1 | κ vs exec | FP on applying-but-failing | null rate |

- [ ] **5.2** Table B, selection accuracy over tasks with ≥1 passing sample:

      | Selector | tasks resolved / 14 | on mixed tasks | 95% CI (bootstrap over tasks) |
      | pass@1 (greedy) | | | |
      | random pick (expected) | | | |
      | shortest trajectory | | | |
      | V_free pick | | | |
      | V_graph pick (gold rubric) | | | |
      | V_graph pick (no-gold rubric) | | | |
      | oracle pass@4 | | | |

- [ ] **5.3** Qualitative: for every mixed-outcome task, show the passing and failing
      patches side by side with both judges' verdicts and reasoning. These are the demo's
      Act 3 and the only place a reader can see *why* a judge was wrong.
- [ ] **5.4** Write-up (≤2 pages, `paper/dec2_addendum.md`): setup, both tables, the
      figure, limitations (14 tasks; N=4; one model family; rubric leakage in the gold
      arm; verifiers judge the patch text only, not the trajectory). State the finding
      in whichever direction it falls. If the gold-rubric arm wins and the no-gold arm
      does not, that *is* the finding: the rubric's value came from the answer leak.

---

## Stage 6 — Optional SFT arm (Kaggle GPU, 2 sessions, Nov 20–26)

Run only if Stage 3 produced pass@4 ≥ 1 for the base policy and ≥ 8 GPU hours remain.

- [ ] **6.1** Complete `TRAINING_FIX_PLAN_DEC2.md` Stage 1 remaining boxes and Stage 2
      (chunked masked loss, `MAX_LEN=8192`). Verify peak memory on one step.
- [ ] **6.2** Replace the invented observation strings in `build_real_sft.py` with the
      real formats from Stage 1.2. Rebuild `sft_real_train.jsonl`. Re-run Cell 8B
      (direction gate) and Gate 9.
- [ ] **6.3** Train **grounded-only** (114 trajectories), `lr=1e-4`, 2 epochs. Read the
      Phase 1.2 decoded spans before the 31B loads. This is the cleanest interpretable
      arm and the one the paper's limitations call for.
- [ ] **6.4** Repeat Stage 2 (greedy) and Stage 3 (N=4) with the adapter →
      `experiments/logs/rollouts/sft_grounded/`. Repeat Stage 4 scoring.
- [ ] **6.5** Table 2 now has three honest rows: base/corrected, base pass@4,
      SFT-grounded. Compare. If SFT < base, that is reportable and important.
- [ ] **6.6** **Submit to the leaderboard only if** SFT tests-pass locally ≥ base
      tests-pass. Otherwise submit base + corrected prompt if the rules permit an
      adapter-free submission (Stage 1.8), or do not submit. Do not spend a slot to learn
      what the local eval already told you.

---

## Stage 7 — Packaging and export (Nov 27 – Dec 1)

- [ ] **7.1** `submission.zip` through the ADK validator; confirm the packaged adapter is
      the chosen checkpoint (`train_rft.py` once overwrote the SFT checkpoint in place —
      check the hash, not the path).
- [ ] **7.2** Export for the demo (`PRESENTATION_PLAN.md` Act 3): for 2–3 mixed-outcome
      tasks, a JSON bundle with every rollout's trace, patch, exec verdict, and all judge
      verdicts + reasoning. Place under `demo/traces/`.
- [ ] **7.3** Commit `experiments/logs/rollouts/**/rollouts.jsonl`,
      `verifier_on_rollouts.json`, the addendum, the figure (force-add past `.gitignore`).
- [ ] **7.4** Keep one submission slot in reserve. Final submission by Dec 1, not Dec 2.

---

## Standing rules

1. A sweep whose verdict column is `unknown` is discarded. Reporting over an unknown
   denominator is F5.
2. The judge is the base model. If `/v1/models` lists an adapter, stop.
3. Every rollout's final patch is saved as text. The patch, not the trajectory, is what
   the judges see — say so in the write-up.
4. Mixed-outcome tasks are the experiment. Count them first; if there are none, the
   selection table is empty and the agreement table is the result.
5. No GPU session starts without the 2.1 pre-checks. The gate is the memory.
