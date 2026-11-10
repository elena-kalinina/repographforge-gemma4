# RepoGraphForge — Why Agentic SFT Silently Fails

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Competition: Gemma 4 Developer Agent](https://img.shields.io/badge/Kaggle-Gemma_4_Developer_Agent-orange.svg)](https://www.kaggle.com/competitions/gemma-4-developer-agent)

This repository accompanies **"Why Agentic SFT Silently Fails: A Failure Taxonomy from
Post-Training a 31B Code Agent"** ([`paper/manuscript.md`](paper/manuscript.md)).

We set out to post-train Gemma 4 31B into a repository-level bug-fixing agent, adapting
GraphForge (`arXiv:2609.38923`) to code graphs. Across the campaign the training loss
converged cleanly in every run — in one case from 2.046 to 0.3935 — and end-task
resolution stayed at approximately zero. Leaderboard scores: 0.00, 0.00, 0.00, 0.03,
0.01, 0.00.

**The paper is the post-mortem, not the system.** What this repository offers is a
taxonomy of six ways an agentic post-training pipeline fails *while reporting success*,
and a set of runnable assertions that catch each one.

> **What we claim:** these six defects occurred, we identified their mechanisms, each is
> invisible to a training loss, and each admits a cheap structural assertion.
>
> **What we do not claim:** that code-graph anchoring improves end-task agent
> performance. Resolution never exceeded 1 of 14 held-out tasks. The pipeline below is
> the setting in which the failures occurred, not a validated method.

---

## The failure taxonomy

| # | Failure | Mechanism | Observable signature |
| :--- | :--- | :--- | :--- |
| **F1** | Stop-token starvation | Masking loop clears the label on the turn-terminating token | Generation runs to `max_output_tokens`; tool calls emitted as prose |
| **F2** | Silent preamble fallback | `try/except` around the tool-schema render fails open | Training sequences lack tool declarations; no error raised |
| **F3** | Inter-stage format drift | RFT filter accepts markdown fences, selecting against tool calls | Submission rate falls *between* stages |
| **F4** | Serialization fragmentation | Four incompatible tool-call encodings in one pipeline | Well-formed calls that nothing parses |
| **F5** | Unfalsifiable metrics | Audit counts attempts, not resolutions | 85.7% reported where truth is 0.0% |
| **F6** | Interface transcription drift | Hand-copied tool declarations diverge from the harness's real parameters | Well-formed calls the sandbox rejects; patches submitted, none resolved |

F6 is the one that cost the most and surfaced last, because **F2 masked it**: declarations
cannot be wrong while they are absent. Five of nine declared tools diverged from the
harness — including `submit_patch`, which takes *no arguments* while every trajectory in
the training set supplied one. Silent failures queue; repairing one exposes the next.

---

## The contribution: structural gates

The generalisable claim is that agentic post-training needs assertions on *structure*,
because the loss observes none of it. Nine gates are stated in §8 of the paper. Two are
implemented here as runnable, GPU-free tests:

```bash
python3 tests/test_chat_format_masking.py   # Gates 1-3: label coverage, stop token, leakage
python3 tests/test_tool_schema_gate.py      # Gate 8: interface identity against the harness
```

Both run offline in under a second. No GPU, no model weights, no Kaggle session.

- [`tests/test_chat_format_masking.py`](tests/test_chat_format_masking.py) replays the
  real measured chat template through the masking routine and asserts the stop token is
  supervised while system turns, user turns and folded tool observations are not.
- [`tests/test_tool_schema_gate.py`](tests/test_tool_schema_gate.py) covers the gate that
  refuses to start training on a missing, stale, or mismatched tool-schema verification —
  including that an operator acknowledgement cannot override a real mismatch.

Both gates live in [`src/chat_format.py`](src/chat_format.py) and are called by
`train_sft.py` before the model loads, so there is no path to training that bypasses them:

- `assert_tool_schema_verified()` (Gate 8) reads a receipt carrying the formatter version,
  a hash of the declared schemas, and the verification outcome.
- `assert_trajectories_conform()` (Gate 9) checks that the calls the **data demonstrates**
  use only parameters the **declarations** contain. These drifted independently in our
  campaign: the schema check went green while every trajectory still demonstrated a call
  the harness rejects.

---

## Repository structure

```text
repographforge-gemma4/
├── paper/
│   ├── manuscript.md             # The paper: failure taxonomy F1-F6, gates, limitations
│   ├── REFRAME.md                # The reframing rationale, with retractions recorded
│   └── manuscript_v1_backup.md   # Pre-reframe draft, kept for provenance
├── src/
│   ├── chat_format.py            # Canonical formatter + masking + the Gate 8 receipt
│   ├── build_real_sft.py         # Grounded trajectories from 115 real maintainer fixes
│   ├── teacher_rollouts.py       # Synthetic teacher rollouts (produces sft_scaled.jsonl)
│   ├── rft_sampler.py            # Best-of-N sampling and rubric filtering
│   ├── graph_sampler.py          # 2-hop ego-graph + cosine embedding sampler
│   ├── task_synthesizer.py       # Reverse fault injection task generator
│   ├── rubric_generator.py       # Evidence rubric (R_G) synthesizer
│   ├── eval_verifier_alignment.py# Table 1 evaluator
│   └── train_qlora.py            # Legacy trainer; the live path is notebook Cell 6
├── tests/                        # The offline gate tests described above
├── notebooks/
│   ├── repographforge_clean_master.md      # Training and evaluation cells
│   └── repographforge_data_preparation.md  # Dataset construction cells
├── experiments/logs/             # Evaluation logs backing the reported numbers
└── submission/                   # Kaggle evaluation package (agent.yaml, adapters/)
```

**Note on the notebooks.** They are Markdown mirrors of Kaggle notebooks. The cell bodies
executed on Kaggle are separate copies and *have* drifted from this repository three times
during development, at least once silently reinstating a defect that had already been
fixed here. Cells carry a version banner and
`chat_format.py` carries `CHAT_FORMAT_VERSION`; a gate refuses to proceed on a mismatch.
This is F6's lesson applied to the development process itself.

---

## The one surviving positive result

A verifier-alignment study over **static candidate patches** — not agent rollouts — so
none of F1–F6 can affect it. Gemma 4 31B as judge, against deterministic test execution.

| Verifier regime | Precision | Recall | F1 | Cohen's κ |
| :--- | :---: | :---: | :---: | :---: |
| Unanchored ($V_{\text{free}}$) | **0.8660** | 0.6512 | 0.7434 | 0.6401 |
| Rubric-anchored ($V_{\text{graph}}$) | 0.8571 | **0.7442** | **0.7967** | **0.7051** |

> **Provisional.** Pending a re-run with task-specific negatives and a confirmed base-model
> judge, reported stratified by negative class. Negatives are synthetic, not harvested from
> real failed rollouts. See §6 and §9 of the paper for the caveats in full.

---

## Reproduction

The diagnostic harness (the tests above) reproduces offline in seconds and is the part we
suggest you actually run. The full pipeline requires the competition dataset, the
`swegemma` wheelhouse and 4× L4-class GPUs.

```bash
git clone <this repo> repographforge-gemma4 && cd repographforge-gemma4
python3 tests/test_chat_format_masking.py
python3 tests/test_tool_schema_gate.py
```

For the full pipeline, run the notebooks in order — data preparation first, then master.
[`RUNBOOK_NOV11.md`](RUNBOOK_NOV11.md) gives the exact cell sequence, prerequisites and
expected output, including what each gate prints when it refuses.

---

## Document index

| Document | Contents |
| :--- | :--- |
| [`paper/manuscript.md`](paper/manuscript.md) | The paper |
| [`RUNBOOK_NOV11.md`](RUNBOOK_NOV11.md) | Ordered execution plan through submission |
| [`DIAGNOSIS.md`](DIAGNOSIS.md) | Original 13-defect diagnosis |
| [`STATUS_REVIEW.md`](STATUS_REVIEW.md) | Post-fix audit, **including a retracted section** |
| [`PAPER_FINALIZATION.md`](PAPER_FINALIZATION.md) | Paper checklist |
| [`TRAINING_FIX_PLAN_DEC2.md`](TRAINING_FIX_PLAN_DEC2.md) | Remaining training work |
| [`PROGRESS_LOG.md`](PROGRESS_LOG.md), [`competition_plan.md`](competition_plan.md) | Historical logs, written before the reframe. Both now carry a superseded banner and inline `[RETRACTED]` corrections on the specific claims the paper withdraws. Kept otherwise unedited — a log quietly rewritten after the fact is worth nothing. |

**On `sft_verified.jsonl`.** The filename is a holdover and is misleading: the admission
filter was `len(messages) >= 4`. Nothing in it was execution-verified. We have left the
name in place rather than rewrite history, and say so here because the paper's §5.5 is
about exactly this kind of unfalsifiable label.

---

## Citation & acknowledgments

Builds on Google DeepMind's Gemma 4 model family, the Kaggle Gemma 4 Developer Agent
benchmark harness (`swegemma`), and GraphForge (`arXiv:2609.38923`).
