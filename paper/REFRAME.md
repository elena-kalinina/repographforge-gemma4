# Paper Reframe — From "Our Agent Works" to "Here Is Why Agentic SFT Silently Fails"

Status: proposed framing for the Nov 12 Paper Track submission.
Companion checklists: `../PAPER_FINALIZATION.md`, `../TRAINING_FIX_PLAN_DEC2.md`.

---

## 1. Why reframe

The drafted thesis is that graph-anchored trajectories plus evidence rubrics produce a
capable offline SWE agent. The measured Pass@1 across every checkpoint is **0.00–0.01**.
That thesis will not be supported by Thursday, and a reviewer who opens Table 2 sees a
zero sitting next to a methodological note conceding the earlier numbers came from a
broken metric.

You do, however, have something rarer and more useful: a complete, mechanistically
explained account of **five silent failures in agentic post-training**, each with a
diagnosis, a fix, and an observable signature. Nobody publishes this, because it only
becomes visible after you have burned a month on it. You have burned the month.

The unifying claim is strong, true, and immediately useful to anyone post-training a
tool-using model:

> **Every one of these failures is invisible to the training loss.** The SFT v1 run
> converged from 2.046 to 0.3935 while training a policy that was structurally incapable
> of ending its turn. Scalar loss cannot see which tokens carry gradient, what fraction of
> the sequence is supervised, whether the stop token appears in the labels, or how many
> examples survived preprocessing. Agentic post-training therefore needs *structural*
> assertions, not just loss curves.

That is a methods contribution. It is checkable, it generalizes beyond Gemma, and it is
honest about the Pass@1.

---

## 2. Proposed title and abstract

> **Why Agentic SFT Silently Fails: A Failure Taxonomy from Post-Training a 31B Code Agent**

> Supervised fine-tuning of tool-using agents is usually monitored with a scalar loss
> curve. We report a post-training campaign on Gemma 4 31B for repository-level bug fixing
> in which the loss converged cleanly in every run while end-task Pass@1 remained at
> approximately zero, and we trace this gap to five distinct defects that are structurally
> invisible to the loss. These are: (F1) masking the end-of-turn token, which removes all
> gradient on the stop decision and yields a policy that cannot terminate a turn; (F2)
> interaction between an evaluation-matched prompt preamble and a drop-on-overflow length
> policy, which can silently reshape or delete the training set; (F3) format drift between
> SFT and a subsequent rejection-sampling stage, where the later stage's accepted
> trajectories contained no tool calls and erased the tool-calling behaviour the earlier
> stage taught; (F4) fragmentation of tool-call serialization across a single pipeline, so
> that the trained format differed from the parsed format; and (F5) evaluation metrics
> that counted submission attempts rather than resolved tasks, inflating apparent
> performance from 0.0% to 85.7%. For each we give the mechanism, the observable
> signature, and the corrective assertion. We additionally report a controlled verifier
> study (n = 387 candidate patches) showing that anchoring an LLM judge to structural
> code-graph contracts improves agreement with deterministic test execution over an
> unanchored judge. We argue that agentic post-training pipelines require structural
> gates — on label coverage, stop-token supervision, dataset survival, and inter-stage
> format identity — and we release the diagnostic harness.

Adjust the F2 wording once the §4 check below is resolved; see `PAPER_FINALIZATION.md` 1.2.

---

## 3. The taxonomy (the paper's spine)

| # | Failure | Mechanism | Observable signature | Corrective assertion |
|---|---|---|---|---|
| **F1** | **Stop-token starvation** | Masking loop set `in_assistant_turn = False` on the turn-close token *before* the mask check, so `labels[i] = -100` landed on the very token that ends the turn. Zero gradient on "stop". | Generations run to `max_output_tokens`; tool calls emitted as conversational prose; turn budget exhausted without `submit_patch`. | `assert eot_id in [l for l in labels if l != -100]` |
| **F2** | **Preamble–budget interaction** | Matching train-time context to eval time adds a large fixed preamble (system prompt + tool schemas). Combined with drop-on-overflow, the surviving set is silently reshaped toward short trajectories. | Dataset size changes with no error; length distribution skews short. | Log kept/dropped; `assert len(kept) >= floor`; report preamble as a fraction of budget. |
| **F3** | **Inter-stage format drift** | The rejection-sampling stage extracted patches with a markdown-fence regex, so accepted trajectories contained zero tool calls. Running last, it overwrote SFT's tool-calling. | End-task submission rate falls *between* stages while loss looks fine. | Assert accepted trajectories contain ≥1 structured tool call before training on them. |
| **F4** | **Serialization fragmentation** | Four mutually incompatible tool-call encodings coexisted in one repository; at most one matched the serving parser. | Model emits well-formed calls in a format nothing parses. | One shared renderer; byte-compare a rendered training prefix against a captured eval prompt. |
| **F5** | **Unfalsifiable metrics** | The audit counted any zero-argument tool call as a submission and never executed tests. | Reported 71.4%–85.7% submission while true resolve rate was 0.0%. | Report applies-cleanly and tests-pass, never attempt counts. |

Each gets a subsection: mechanism → the code → the signature → the fix → the assertion.

---

## 4. One open question that changes F2's wording

**RE-RESOLVED 2026-11-11 — the 2026-11-10 answer below was wrong.** It argued the
preamble (1,110 tokens, measured) was absent because the median trajectory was 831 tokens.
That median was measured on raw message JSON, which never contains the preamble, so the
comparison establishes nothing. Correct status: reading (a) holds **by construction** for
runs 1–5 (the original code passed no `tools=` at all, DIAGNOSIS C2); for runs 6–7, which
used the fail-open renderer, the question is **undetermined** — neither reading (a) nor
(b) can be chosen from the records. Manuscript §5.2 now states it both ways, explicitly,
with the retracted arithmetic named.

Superseded 2026-11-10 wording, kept for the record: *"The median trajectory was 831
tokens, so at least half the training set was shorter than the preamble alone, which is
impossible if it was present. The `tools=` render had been failing silently."*

Original wording, kept for the record:

`src/chat_format.py` renders tool schemas inside a `try/except Exception: pass` that falls
through to a no-tools render. The training log reports **299 kept, 5 dropped at
MAX_LEN=2560**. That many trajectories fitting is hard to reconcile with the preamble
actually being present.

Two readings, both consequential, both cheap to settle:

- **(a) The `tools=` render silently failed.** Then C2 was only half-applied — the system
  prompt is in the training text but the tool declarations are not — and train/eval
  mismatch on tool schemas persists. F2 becomes "the silent fallback", which is arguably a
  *better* story: the pipeline reported success while dropping the fix on the floor.
- **(b) The template accepted tools and the lengths genuinely fit.** Then F2 is a
  near-miss rather than a realized failure, and it should be reported as a hazard
  demonstrated by construction, not as something that bit this run.

Resolve with the three-line check in `PAPER_FINALIZATION.md` 1.2 before writing §F2.
Do not write F2 both ways and pick later — the wording differs materially.

---

## 5. What survives, what changes, what goes

### Survives, promote it
- **Table 1 (verifier alignment).** This is a *judging* experiment on static candidate
  patches. It is methodologically independent of every agent failure above, so it is the
  one quantitative positive result that stands. It must be re-run with task-specific
  negatives (`candidate_patch_generator.py`, already implemented) and a base-model judge
  (already implemented), and reported stratified by negative class.
- **Table 2 as taxonomy evidence.** Keep the checkpoint-by-checkpoint table. Reframed, the
  zeros are the finding rather than an embarrassment.
- **The code-graph navigation apparatus.** Subgraph sampling, embedding neighbours, the
  rubric formalism. Present as system description and released artifact, not as a result.

### Changes framing
- RQ1–RQ4 become **F1–F5 plus the verifier study**. The research question is no longer
  "does graph anchoring help an agent" but "why did a cleanly converging pipeline produce
  a non-functional agent, and what must be asserted to catch it".
- Case studies (`requests_7505`, `requests_6644`, `requests_7433`) stay, explicitly
  labelled **qualitative, n = 1**, illustrating that capability was present but
  unreliable — which is itself evidence for the variance argument.

### Must come out
- Any description of trajectories as "verified" or "pristine" — the filter was `len(msgs) >= 4`.
- The 2× `V_graph` acceptance gain, unless re-run with the base-model judge; as it stands
  the policy graded its own rollouts.
- Pass@1 or capability claims resting on the n = 1 successes.
- Table 3 / multi-scale edge-hardware claims that were never run. Move to Future Work.
- "71.4%" and "85.7%" as performance, anywhere except inside F5 as the error being reported.

---

## 6. Section outline

1. **Introduction** — loss converged, Pass@1 ≈ 0; contribution is the taxonomy + assertions.
2. **Background** — agentic SFT, tool-call formats, SWE-bench-style evaluation.
3. **System** — RepoGraphForge: graphs, subgraph sampling, evidence rubrics, trajectory
   synthesis, SFT→RFT. Describe honestly, including which parts were unverified.
4. **Failure taxonomy** — §F1–F5, one subsection each, mechanism → signature → fix → assertion.
5. **Verifier study** — Table 1, re-run, stratified. The positive result.
6. **Evidence** — Table 2 checkpoint progression; F1 ablation if it runs; case studies.
7. **Structural gates for agentic post-training** — the generalizable prescription, as a
   checklist a reader can apply to their own pipeline. This is the takeaway section.
8. **Limitations** — single model family, single task family, Pass@1 ≈ 0 means no claim
   about whether graph anchoring helps end-task performance. Say it plainly.
9. **Future work** — Dec 2 leaderboard plan, multi-scale profiling.

---

## 7. The one experiment worth GPU time

F1 is the headline and currently rests on *observational* evidence: the manuscript's own
Table 2 rows (SFT v1 with the masked stop token vs. later runs) differ in several ways at
once, so they do not isolate the cause.

A controlled ablation would: train two adapters identical in every respect except that one
masks the end-of-turn token, then report **% of responses hitting `max_output_tokens`**
and **strict `submit_patch` rate**. Even at small scale — a few dozen trajectories, a
handful of eval tasks — this is a clean controlled result and it is the heart of the paper.

If quota or time does not allow it before Nov 12: report F1 observationally, state
explicitly that it is not a controlled ablation, and list it as the primary planned
experiment. That is honest and still publishable. Do not describe an ablation you did not run.
