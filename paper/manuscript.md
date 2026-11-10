# Why Agentic SFT Silently Fails: A Failure Taxonomy from Post-Training a 31B Code Agent

**Author:** Elena Kalinina
**Affiliation:** Independent Research / Kaggle Gemma 4 Developer Agent Competition
**Track:** Google DeepMind — The Gemma 4 Developer Agent Paper Track
**Date:** November 2026

---

## Abstract

Supervised fine-tuning of tool-using agents is usually monitored with a scalar training loss. We report a post-training campaign on Gemma 4 31B for repository-level bug fixing in which the loss converged cleanly in every run — in one case from 2.046 to 0.3935 — while end-task resolution remained at approximately zero throughout. We trace this gap to five distinct defects, each of which is structurally invisible to a loss curve.

These are: **(F1) stop-token starvation**, in which the end-of-turn token receives no gradient because a masking loop clears its label, producing a policy that cannot terminate a turn and emits tool calls as conversational prose; **(F2) silent preamble fallback**, in which the code path that aligns training context with evaluation context fails open, so tool declarations are absent from every training sequence while the pipeline reports success; **(F3) inter-stage format drift**, in which a rejection-sampling stage accepted only markdown-fenced patches, contained zero tool calls, and — running last — erased the tool-calling behaviour the preceding stage had taught; **(F4) serialization fragmentation**, in which four mutually incompatible tool-call encodings coexisted in one pipeline, none of which matched the serving template; and **(F5) unfalsifiable metrics**, in which the audit counted submission attempts rather than resolved tasks, reporting 85.7% where the true resolution rate was 0.0%.

For each failure we give the mechanism, the code that produces it, its observable signature, and a corrective assertion. We additionally report a controlled verifier study over candidate patches indicating that anchoring an LLM judge to structural code-graph contracts improves agreement with deterministic test execution relative to an unanchored judge. We make no claim that code-graph anchoring improves end-task agent performance; our resolution rates are too close to zero to support one. Our contribution is diagnostic: agentic post-training pipelines require **structural gates** — on label coverage, stop-token supervision, dataset survival, and inter-stage format identity — because scalar loss cannot observe any of them. We release the diagnostic harness.

---

## 1. Introduction

Open-weights models such as Google's Gemma 4 family make locally hosted, privacy-preserving software-engineering agents feasible for organisations that cannot transmit source code to external APIs. Adapting such a model into a competent repository agent is ordinarily framed as a data and objective problem: collect trajectories, fine-tune, measure the loss, evaluate.

We undertook exactly this. Over five post-training runs on Gemma 4 31B — supervised fine-tuning on code-graph-anchored trajectories, followed by best-of-$N$ rejection fine-tuning filtered by structural rubrics — every run produced a smooth, converging loss curve. Every run also produced an agent that resolved approximately zero held-out tasks. The leaderboard scores across the campaign were 0.00, 0.00, 0.00, 0.03, 0.01, and 0.00.

The central observation of this paper is that **the loss curve was never going to reveal why.** A cross-entropy loss over a token sequence is a scalar summary. It cannot report which tokens carried gradient, what fraction of the sequence was supervised, whether the token that terminates a turn appeared in the labels at all, whether the tool declarations the evaluation harness sends were present during training, or how many training examples survived preprocessing. Each of those was wrong at some point in our campaign, and each was wrong *silently*: the pipeline logged success, the loss fell, and the agent did not work.

This paper is the post-mortem. We present five failures as a taxonomy (§5), each with mechanism, signature, fix, and a runnable assertion that would have caught it. We report the one experiment that survives the campaign as a positive result — a verifier-alignment study (§6) that is methodologically independent of the agent failures because it evaluates static patches rather than agent rollouts. We then generalise the taxonomy into a set of structural gates for agentic post-training (§8), which we regard as the paper's practical contribution.

### 1.1 What this paper does and does not claim

We are explicit about the boundary, because the original framing of this work claimed more.

**We claim:** that these five defects occurred, that we have identified their mechanisms, that each is invisible to the training loss, and that each admits a cheap structural assertion. We claim the verifier-alignment result of §6 within its stated limitations.

**We do not claim:** that code-graph anchoring improves end-task agent performance. Our resolution rate never exceeded 1 of 14 held-out tasks. Any such claim would require a working agent as a baseline, which we do not have. The system described in §3 should be read as the setting in which the failures occurred, not as a validated method.

### 1.2 Contributions

1. **A taxonomy of five silent failures in agentic post-training** (§5), each with mechanism, observable signature, corrective fix, and a runnable assertion.
2. **An environment-grounded account of tool-call serialization drift** (§5.4, Appendix A), including a dump of the real Gemma 4 chat template that contradicts the format our own earlier methodology section documented.
3. **A verifier-alignment study** (§6) comparing unanchored and rubric-anchored LLM judges against deterministic test execution, reported with its contamination caveats stated.
4. **A set of structural gates for agentic post-training pipelines** (§8), stated as assertions a practitioner can paste into their own trainer.
5. **A released diagnostic harness** (`src/chat_format.py` and the verification gate) implementing those assertions.

---

## 2. Background

**Agentic supervised fine-tuning.** Training a model to use tools means training it to emit structured calls, consume environment observations, and terminate turns so the harness can execute the call. The target distribution therefore includes control tokens whose semantics are defined by the serving stack, not by the data. This differs from ordinary instruction tuning, where the model's output is consumed as text and the only structural requirement is to stop at the end.

**Rejection fine-tuning.** Sampling $N$ rollouts per task, filtering with a verifier, and fine-tuning on the survivors is a standard variance-reduction step. Because it runs *after* supervised fine-tuning, its data distribution overwrites the earlier stage's. If the filter's acceptance criterion is not expressed in the same output format the earlier stage taught, the later stage will systematically select against that format (§5.3).

**LLM-as-a-judge and verification.** Running full test suites is slow and limited by coverage, so LLM judges are commonly substituted. Unanchored judges are known to be unreliable. Anchoring them to explicit criteria is the motivation for the rubric study in §6.

**Silent failure.** The failures in this paper share a property: a defective pipeline and a correct pipeline produce *indistinguishable training loss curves*. This is not specific to our setting. Any pipeline in which preprocessing, masking, and serving format are separate from the objective can exhibit it.

---

## 3. System Description

This section describes the pipeline in which the failures occurred. It is a system description, not a set of validated results. Where a component was never verified, we say so.

### 3.1 Repository multigraph ingestion and ego-subgraph sampling

The competition provides pre-computed repository multigraphs $G = (V, E)$, where vertices correspond to modules, classes, methods, and functions, and directed multigraph edges capture $k \in \{\text{imports}, \text{calls}, \text{inherits}, \text{overrides}\}$. Each node $v$ carries a 256-dimensional embedding $\mathbf{z}_v \in \mathbb{R}^{256}$.

`src/graph_sampler.py` samples an anchor node $v_0$ and extracts a connected subgraph $G_S = (V_S, E_S)$ combining:

1. **Structural 2-hop ego-graph:** $V_{\text{struct}} = \{v \in V \mid \text{dist}_G(v_0, v) \le 2\}$.
2. **Semantic embedding expansion:** $V_{\text{semantic}} = \operatorname{arg\,top\,k}_{u \in V \setminus V_{\text{struct}}} \cos(\mathbf{z}_{v_0}, \mathbf{z}_u)$, capturing couplings not expressed as imports.

The union $V_S = V_{\text{struct}} \cup V_{\text{semantic}}$ induces $G_S$, spanning 3–8 nodes across 1–3 source files. We sampled 1,139 such subgraphs across `fastapi/fastapi` and `Textualize/rich`.

### 3.2 Task synthesis and evidence rubric compilation

Two task sources were used, and the distinction matters for interpreting every result below.

**Synthetic tasks (reverse fault injection).** A teacher model is shown a subgraph and asked to introduce a cross-node contract violation, producing a bug patch $\Delta_{\text{bug}}$, a reproduction test, and an evidence rubric. The intended ground-truth fix is the inversion $\Delta_{\text{gold}} = -\Delta_{\text{bug}}$.

*Verification status: none.* The generated patches were never applied to a repository, the reproduction tests were never executed, and the trajectory admission filter was a length check (`len(messages) >= 4`). The synthesiser was additionally prompted with node *names* only, never source text, so generated diffs reference invented context lines. An early version of the pipeline also passed $\Delta_{\text{bug}}$ to the trajectory teacher under the label "Reference Fix Patch", inverting the training target (§5.6).

**Grounded tasks.** `src/build_real_sft.py` constructs trajectories from the 115 real training tasks, whose `patch` field is the maintainer's actual fix. Real graph nodes populate the subgraph turn; exact pre-fix and post-fix hunk text populate the `read_file` and `edit_file` turns.

*Verification status: patch direction checked; observations synthetic.* The observation strings are constructed to a plausible format rather than captured from the live tool (§5.4).

**Evidence rubrics ($R_G$).** For each task, a structured checklist of 3–5 criteria anchored to node identifiers in $G_S$: target symbol invariants, interface and type contracts for caller nodes $N_{\text{in}}(G_S)$, and exception/regression invariants.

### 3.3 The three verifier regimes

1. **Unanchored judge $V_{\text{free}}(x, \hat{y}) \in \{\text{PASS}, \text{FAIL}\}$** — issue description and candidate patch only.
2. **Rubric judge $V_{\text{graph}}(x, \hat{y}, R_G)$** — PASS iff $\forall c_i \in R_G,\ \hat{y} \models c_i$.
3. **Execution oracle $V_{\text{exec}}$** — the official `test_patch` under `pytest` in an isolated sandbox; exit code 0 is PASS.

### 3.4 Two-stage post-training

**Stage A (SFT).** Rank-32 LoRA ($r=32$, $\alpha=64$) across the text transformer's linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), AdamW, cosine decay, warmup ratio 0.05, effective batch size 8. Learning rate $2 \times 10^{-4}$ for 3 epochs in early runs, reduced to $1 \times 10^{-4}$ for 2 epochs after the campaign showed the higher rate was sufficient to overwrite base instruction-following on a dataset of a few hundred examples.

The intended objective is cross-entropy over assistant tokens only:
\[
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t \in \mathcal{T}_{\text{assistant}}} \log \pi_\theta(y_t \mid y_{<t}, x)
\]
The realisation of $\mathcal{T}_{\text{assistant}}$ is where F1 occurs (§5.1).

**Stage B (RFT).** For each task, sample $N=4$ trajectories at $T=0.7$ from the SFT checkpoint, score each with $V_{\text{graph}}$, select the shortest passing trajectory $\tau^*_i = \operatorname{arg\,min}_{\tau_{i,j} \mid v_{i,j}=1} |\tau_{i,j}|$, and fine-tune the SFT adapter at $\eta = 5 \times 10^{-5}$. The realisation of the acceptance criterion is where F3 occurs (§5.3).

---

## 4. Experimental Setup

### 4.1 Anti-leakage partition

| Split | Repositories | Tasks | Role |
| :--- | :--- | :---: | :--- |
| Training ($D_{\text{train}}$) | `fastapi/fastapi`, `Textualize/rich` | 115 | Subgraph sampling, task synthesis, trajectory collection |
| OOD holdout ($D_{\text{val-ood}}$) | `psf/requests`, `encode/httpx` | 14 | Held-out evaluation |

Partitioning is strictly by repository, so holdout repositories contribute no graphs, no synthesised tasks, and no trajectories.

### 4.2 Training data composition

The final SFT dataset (`sft_combined.jsonl`) contained **299 trajectories after preprocessing**: 115 grounded trajectories from real maintainer fixes and approximately 184 synthetic teacher trajectories. **Roughly 62% of the training signal was therefore unverified synthetic data** with hallucinated environment observations. This is a material limitation (§9) and we report it wherever a training result is discussed.

### 4.3 Hardware and environment

All training, generation, and evaluation ran offline on 4× NVIDIA L4 (96 GB total) with `transformers 5.13.1` and vLLM at tensor parallelism $tp=4$, in a sandbox with no network access.

### 4.4 Environment facts established before analysis

Several conclusions depend on properties of the serving stack rather than our code. We established these by direct measurement before analysing any failure; the full dump is in Appendix A. The load-bearing facts:

- The chat template emits `<bos>` itself, and `add_special_tokens=True` does **not** add a second one. A double-BOS hypothesis we had held was therefore **false**, and we record it as such.
- Turn delimiters are `<|turn>` and `<turn|>`; tool calls are delimited `<|tool_call>` … `<tool_call|>` and tool responses `<|tool_response>` … `<tool_response|>`, with `<|"|>` as the string delimiter.
- **Tool responses are folded inside the preceding model turn rather than emitted as their own turn.** This is central to §5.1.
- Trajectory token lengths: median 831, p90 1475, p95 1709, p99 2729, maximum 4367.

---

## 5. The Failure Taxonomy

| # | Failure | Mechanism | Observable signature |
| :--- | :--- | :--- | :--- |
| **F1** | Stop-token starvation | Masking loop clears the label on the turn-terminating token | Generation runs to `max_output_tokens`; tool calls emitted as prose |
| **F2** | Silent preamble fallback | `try/except` around the tool-schema render fails open | Training sequences lack tool declarations; no error raised |
| **F3** | Inter-stage format drift | RFT filter accepts markdown fences, selecting against tool calls | Submission rate falls *between* stages |
| **F4** | Serialization fragmentation | Four incompatible tool-call encodings in one pipeline | Well-formed calls that nothing parses |
| **F5** | Unfalsifiable metrics | Audit counts attempts, not resolutions | 85.7% reported where truth is 0.0% |

A sixth defect, label-direction inversion, is reported in §5.6; it is a data-pipeline error rather than a silent-failure mode, since it is detectable by applying the label.

### 5.1 F1 — Stop-token starvation

**Mechanism.** The intended objective supervises assistant tokens and masks everything else. The implementation walked the token sequence maintaining an `in_assistant_turn` flag:

```python
in_assistant_turn = False
for i in range(len(input_ids)):
    if input_ids[i] == model_turn_id:                           # <|turn>
        in_assistant_turn = True
    elif input_ids[i] == turn_end_id and in_assistant_turn:     # <turn|>
        in_assistant_turn = False
    if not in_assistant_turn:
        labels[i] = -100
```

Two defects. First, `<|turn>` is the generic turn *opener*; the role token follows it. The flag is therefore set at the start of every turn, including user turns, so the supposed assistant-only masking supervises the entire sequence — including environment observations, teaching the model to generate its own tool outputs.

Second, and decisively: when `input_ids[i]` is the turn-terminating token, the flag is cleared *before* the mask check, so `labels[i] = -100` lands on exactly the token that ends the turn. Across three epochs the model received **zero gradient on the decision to stop**.

**Signature.** A policy that cannot terminate a turn does not fail loudly. It generates until `max_output_tokens`, so the harness receives an unterminated blob rather than a parseable call. Downstream this appears as tool calls rendered as conversational prose, exhausted turn budgets, and completion without `submit_patch`. Our campaign logged all three and initially attributed them to a separate cause — the absence of structural token wrapping — which was incorrect.

**Why the loss did not show it.** The stop token is one position in a sequence of hundreds. Masking it changes the loss by a negligible amount. The SFT v1 run converged 2.046 → 0.3935 while training a policy structurally incapable of ending a turn.

**Fix.** Derive spans from the chat template rather than inferring them, and include the terminating token in the supervised span. Our implementation renders each message prefix incrementally and takes the token difference, falling back to a span-marker walk when the template rewrites earlier tokens — which, as §4.4 notes, it does whenever a tool response is folded into a model turn.

**Assertion.**
```python
eot_id = tokenizer.convert_tokens_to_ids("<turn|>")
assert eot_id in [l for l in labels if l != -100], "stop token is not supervised"
```

### 5.2 F2 — Silent preamble fallback

**Mechanism.** To make training context match evaluation context, training sequences should carry the same system prompt and the same tool declarations the harness sends. Our renderer attempted this:

```python
try:
    if tools:
        return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, ...)
except Exception:
    pass
return tokenizer.apply_chat_template(msgs, tokenize=False, ...)   # no tools
```

The `except Exception: pass` makes the function **fail open**. If the template does not accept a `tools` argument, every training sequence is rendered without tool declarations and nothing reports it.

**Signature, and how we detected it.** The detection is arithmetic rather than instrumented, which is itself the point. The tool schemas measure approximately 1,473 tokens and the system prompt approximately 357, a fixed preamble of roughly 1,830 tokens per example. The measured **median** trajectory length was 831 tokens (§4.4) — below the preamble alone. The preamble was therefore not present, and the fallback had been taken silently on every example.

This also explains an observation that would otherwise be puzzling: at a 2,560-token cap, only **5 of 304** trajectories were dropped. With the preamble present, the p90 of 1,475 would have become roughly 3,300 and the majority would have been dropped.

**Interaction with length policy.** The pipeline used drop-on-overflow rather than truncation, on the correct reasoning that right-truncation silently deletes the `edit_file` and `submit_patch` turns that constitute the label. Drop-on-overflow is safe at a generous budget. Combined with a large fixed preamble and a reduced budget, it becomes a dataset filter that silently reshapes the training distribution toward short trajectories. We did not realise this failure mode — the preamble was absent, so almost nothing was dropped — but we report it because the combination is a live hazard in any pipeline that matches eval-time context and caps sequence length.

**Fix.** Fail closed. Assert the rendered text contains the declarations. Log kept/dropped counts and enforce a floor.

**Assertion.**
```python
rendered = render(msgs, tokenizer, tools=TOOLS)
assert "old_string" in rendered and "node_id" in rendered, "tool schemas missing from training text"
assert len(kept) >= FLOOR, f"only {len(kept)} trajectories survived preprocessing"
```

### 5.3 F3 — Inter-stage format drift

**Mechanism.** Stage B samples rollouts, filters them, and fine-tunes on survivors. Patch extraction from a rollout was implemented as:

```python
m = re.search(r"```(?:diff|patch)?\n([\s\S]*?)```", text)
```

A rollout qualifies only if it contains a markdown-fenced diff. A rollout that correctly invokes `edit_file` and `submit_patch` produces **no fenced block** and is rejected. The accepted set therefore contained zero tool calls by construction.

Because Stage B runs *after* Stage A and writes to the same adapter, the final policy was fine-tuned on a distribution that systematically excluded the behaviour Stage A existed to teach.

**Signature.** The giveaway is a submission rate that falls *between* stages while both stages' losses look healthy. Holdout submissions fell from 4/14 to 1/14 across the RFT pass.

**Compounding factors.** The filter accepted 4 of 50 tasks, so Stage B trained on 4 examples — approximately 4 optimizer steps at the configured accumulation — with gradient norm falling from 239.7 to 7.29 in 29 seconds. The judge was additionally the policy adapter itself rather than the base model, so the filter was contaminated.

**Fix.** Extract patches from `edit_file` tool-call arguments. Require a minimum accepted count before training. Use the unadapted base model as judge.

**Assertion.**
```python
assert all(has_tool_call(t) for t in accepted), "accepted trajectories contain no tool calls"
assert len(accepted) >= 50, f"only {len(accepted)} accepted; refusing to run RFT"
```

### 5.4 F4 — Serialization fragmentation

**Mechanism.** Four mutually incompatible tool-call encodings coexisted in one repository:

| Producer | Encoding |
| :--- | :--- |
| `teacher_rollouts.py` | `<\|tool_call\|>call:name{args}<\|tool_call\|>` inside `<start_of_turn>model` |
| `rft_sampler.py` | `<thought>…</thought><tool_call>{"name":…}</tool_call>` |
| SFT notebook cell | native `tool_calls` dicts, `arguments` as **dict** |
| `train_qlora.py` | native `tool_calls` dicts, `arguments` as **JSON string** |
| RFT notebook cell | none — a markdown fence (§5.3) |

At most one can match the serving template. The measured template (§4.4, Appendix A) is:

```
<|turn>model
<|tool_call>call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|>
```

with asymmetric delimiters (`<|tool_call>` opening, `<tool_call|>` closing) and `<|"|>` as the string delimiter. **None of the four hand-written encodings matches it.** Our own methodology section, in an earlier draft of this paper, documented the `<start_of_turn>model` / `<|tool_call|>` form — a format that does not exist in this model. We note this as evidence of how readily such drift propagates from code into prose.

**Fix.** One shared renderer that delegates to the tokenizer's own template; never hand-write the encoding. Import tool schemas from the harness rather than transcribing them. Byte-compare a rendered training prefix against a captured evaluation prompt.

**Assertion.**
```python
assert render(msgs, tok)[:300] == captured_eval_prompt[:300], "train/eval prompt divergence"
```

### 5.5 F5 — Unfalsifiable metrics

**Mechanism.** The audit classified a trace as having submitted a patch via:

```python
elif args == {} or fn.get("name") == "submit_patch":
    submitted = True
```

Any zero-argument tool call — `get_status()` among them — satisfies the first clause. Reported submission rates of 71.4% and 85.7% were inflated by an unknown amount.

More fundamentally, *submission is not resolution*. The audit never applied the diff and never ran a test. A run reporting 6 of 14 submissions and a leaderboard score of 0.00 is entirely consistent: patches were emitted, and none of them worked.

**Consequence.** For most of the campaign, no local measurement could distinguish a good checkpoint from a bad one. Every tuning decision was made against a metric uncorrelated with the objective.

**Fix.** Report applies-cleanly (`git apply --check`) and tests-pass. Never report attempt counts as performance.

**Assertion.**
```python
assert "resolved" in audit_columns, "eval does not measure task resolution"
```

### 5.6 A sixth defect: label-direction inversion

The synthetic task record defines `bug_patch` as *"a unified git diff that introduces the bug into the working code."* The trajectory generator read that field and presented it to the teacher as the answer:

```python
patch = task.get("bug_patch", "")[:2000]
...
f"### Reference Fix Patch:\n{patch}\n\n"
```

Every synthetic trajectory therefore demonstrated applying the defect. We separate this from F1–F5 because it is **not** a silent failure: applying the label and running the test detects it immediately. It went undetected only because nothing in the pipeline executed anything (§3.2) — which makes it a consequence of F5 rather than an independent failure mode.

The fix was to build trajectories from the 115 real maintainer fixes, and to rename the field so the inversion cannot recur:

```python
assert "fix" in field_name, "patch field must be the fix direction"
```

---

## 6. Verifier Alignment Study

This is the one experiment in the campaign that is methodologically independent of the agent failures: it evaluates **static candidate patches**, not agent rollouts, so none of F1–F5 can affect it.

**Setup.** A balanced benchmark of candidate patches across the 129 competition tasks: gold maintainer fixes ($y^+$), truncated contract-omission patches ($y^-$), and sycophantic-drift patches ($y^-$). Each is scored by $V_{\text{free}}$ and $V_{\text{graph}}$ and compared against $V_{\text{exec}}$.

**Pilot results.** *(These numbers come from the pilot benchmark and are reported with the contamination caveat below. They should be regarded as provisional pending the re-run described under Threats.)*

| Verifier regime | Precision | Recall | F1 | FP drift rate | Cohen's $\kappa$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Unanchored ($V_{\text{free}}$) | **0.8660** | 0.6512 | 0.7434 | 0.0506 | 0.6401 |
| Rubric-anchored ($V_{\text{graph}}$) | 0.8571 | **0.7442** | **0.7967** | 0.0623 | **0.7051** |
| Relative $\Delta$ | −1.03% | **+14.28%** | **+7.17%** | +1.17% | **+10.15%** |

**Interpretation.** The unanchored judge falsely rejects roughly a third of working maintainer fixes (recall 0.6512). Anchoring to explicit structural contracts raises recall to 0.7442 at a precision cost of about one point, and raises $\kappa$ from 0.6401 to 0.7051. The direction of the effect — anchoring trades a little precision for materially better recall and better agreement with execution — is the claim we are willing to defend.

**Threats to validity.** Three, stated plainly:

1. **Contaminated negative class.** In the pilot, the `sycophantic_drift` negatives used a single constant three-line string identical across all tasks, making approximately one third of the negative pool trivially separable and inflating both verifiers' scores. `src/candidate_patch_generator.py` now constructs task-specific drift patches targeting each task's actual file and hunk. **The table above predates that change.**
2. **Judge contamination.** The pilot queried the served adapter. The corrected configuration queries the unadapted base model.
3. **Synthetic negatives.** Negatives are constructed, not harvested from real failed rollouts, and may be easier than naturally occurring wrong patches.

A re-run addressing (1) and (2), with false-positive rates reported separately for `partial_contract_omission` and `sycophantic_drift`, is required before this result should be cited. **If the corrected gain is smaller, the smaller number is the result.**

---

## 7. Aggregate Evidence

### 7.1 Checkpoint progression

| Policy variant | Leaderboard | Strict non-empty patch | Pytest resolved | Target file localized | Dominant failure |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 1. Zero-shot 12B | 0.00 | 0.0% (0/14) | 0.0% (0/14) | 14.3% (2/14) | Blind `grep`, timeout |
| 2. Zero-shot 31B | 0.00 | 0.0% (0/14) | 0.0% (0/14) | 21.4% (3/14) | Trial-and-error search |
| 3. SFT v1 | 0.00 | 0.0% (0/14) | 0.0% (0/14) | 71.4% (10/14) | **F1** (stop token) |
| 4. Pilot SFT+RFT | 0.03 | 10/14 *(legacy audit)* | 7.1% (1/14) | 92.9% (13/14) | **F5**; truncation |
| 5. Scaled SFT, no RFT | 0.01 | 28.6% (4/14) | 7.1% (1/14) | 85.7% (12/14) | **F1** residual |
| 6. 4-sample RFT | — | 7.1% (1/14) | 0.0% (0/14) | — | **F3** (fence drift) |
| 7. Grounded SFT | 0.00 | 42.9% (6/14) | 0.0% (0/14) | — | **F2**/**F4** suspected |

**Localization rose from 14.3% to over 85% while resolution never exceeded 1 of 14.** The campaign consistently taught the model *where* the bug was and consistently failed to let it act on that.

**On the 0.03 vs 0.01 comparison.** An earlier draft of this paper read the difference between rows 4 and 5 as evidence that rejection fine-tuning is indispensable, describing it as a "3× gain" and "definitive blind empirical proof". We retract that. On approximately 60 hidden tasks, 0.03 versus 0.01 is a difference of roughly one or two resolved tasks — well within noise, with no confidence interval and no repeated runs. Rows 6 and 7 subsequently scored 0.00 with RFT variants in the pipeline, which contradicts the claimed direction outright. **No conclusion about RFT's value should be drawn from this table.**

### 7.2 The central observation: loss convergence against resolution

Every run in the campaign converged. The SFT v1 run fell 2.046 → 0.3935. The SFT v2 run fell 2.877 → 0.9554 in under 20 minutes. The RFT pass fell 1.79 → 1.083 with gradient norm stabilising 239.7 → 7.29.

Resolution across those same checkpoints: 0/14, 0/14, 0/14, 1/14, 1/14, 0/14, 0/14.

Plotted together — loss on one axis, resolution on the other — these form the paper's central figure, and the argument of §8 in a single image. Loss convergence carried **no information** about whether the policy could terminate a turn (F1), whether it had ever seen the tool declarations it would receive (F2), whether the final stage had erased the behaviour of the first (F3), whether its output format matched the parser (F4), or whether the metric being optimised measured anything (F5).

### 7.3 Qualitative case studies

These are **single observations**, reported as existence proofs that the underlying capability was present and unreliable, not as performance claims.

**`requests_7505`.** The agent localised a proxy-stream defect via the subgraph, modified `src/requests/models.py` at the `RequestEncodingMixin` and `PreparedRequest` sites, and introduced `hasattr(data, "read")` alongside `isinstance(data, _SupportsRead)` — a duck-typing-correct fix matching maintainer intent. Completed in 53 seconds using 8 of 15 tool calls.

**`requests_6644`.** Resolved under `pytest` in 18.9 seconds with 3 tool calls.

**`requests_7502`.** Symptom in `models.py`, implementation in `utils.py`. The agent exhausted its budget paging the caller file. This is the cross-file indirection case that motivated 2-hop semantic expansion; we record it as a qualitative observation only, since with resolution near zero we cannot show that the expansion helps.

That `requests_7505` and `requests_6644` happened at all establishes that capability was induced. That they did not generalise is what F1–F5 explain.

---

## 8. Structural Gates for Agentic Post-Training

The generalisable claim: **agentic post-training pipelines require assertions on structure, because the loss observes none of it.** We propose the following gates, each derived from a failure above and each cheap enough to run before every training job.

**Gate 1 — Label coverage.** Decode the supervised tokens and read them.
```python
unmasked = [i for i, l in zip(input_ids, labels) if l != -100]
frac = len(unmasked) / len(input_ids)
assert 0.03 < frac < 0.45, f"supervised fraction {frac:.1%} out of range"
print(tokenizer.decode(unmasked, skip_special_tokens=False))
```
The printed text must be assistant turns and nothing else. This single check catches F1 and the observation-leakage half of the masking bug. In our campaign it would have saved four training runs.

**Gate 2 — Stop-token supervision.**
```python
assert eot_id in [l for l in labels if l != -100], "stop token unsupervised"
```

**Gate 3 — No observation leakage.**
```python
for marker in ("File: ", '"status": "ok"', "<|tool_response>"):
    assert marker not in tokenizer.decode(unmasked), f"observation leaked: {marker}"
```

**Gate 4 — Train/eval prompt identity.** Render a training example, capture a real evaluation prompt, and byte-compare the prefix. Fail closed: a renderer that falls back silently when tool declarations cannot be attached will hide F2 indefinitely.

**Gate 5 — Dataset survival.** Log kept and dropped counts; assert a floor; assert a minimum optimizer-step count. A pipeline that trains on 4 examples for 4 steps should refuse to start.

**Gate 6 — Inter-stage format identity.** Any stage that consumes the previous stage's outputs must accept the format the previous stage produced. Assert that accepted trajectories contain at least one structured tool call before training on them.

**Gate 7 — Falsifiable evaluation.** Report applies-cleanly and tests-pass. Never report attempt counts as performance. Without this gate, none of the others can be prioritised, because no measurement distinguishes a fix from a regression.

We note that Gates 1–3 are properties of the *tokenized batch*, Gates 4 and 6 of *format identity across boundaries*, and Gates 5 and 7 of *measurement*. None is expensive. All are absent from standard fine-tuning recipes.

---

## 9. Limitations

1. **No end-task claim.** Resolution never exceeded 1 of 14 held-out tasks. We cannot say whether code-graph anchoring improves agent performance; the experiment that would answer it has not been run successfully.
2. **F1 is observational.** Our evidence is the checkpoint progression in §7.1, in which runs differ in more than one respect. A controlled ablation — two adapters identical but for stop-token masking, reporting ceiling-hit rate and submission rate — is the primary planned experiment and **was not run**.
3. **Training data quality.** Approximately 62% of the SFT set was unverified synthetic trajectories with hallucinated observations (§4.2).
4. **Synthetic observations in grounded data.** Even the grounded trajectories use constructed observation strings rather than captured tool output (§3.2).
5. **Verifier study caveats.** Three threats stated in §6; the headline numbers predate two of the three corrections.
6. **Single model family, single task family.** One model, two training repositories, two holdout repositories. We expect the failure *modes* to generalise, since they concern pipeline structure rather than model specifics, but we have not shown this.
7. **No hardware profiling.** An earlier draft reported a multi-scale VRAM and throughput table across Gemma 4 E2B, E4B, 12B, and 31B. Those measurements were never taken and the table has been removed.
8. **Small holdout.** 14 tasks; leaderboard ~60 hidden tasks. Single-task differences move the reported rate by 7 percentage points.

---

## 10. Future Work

**Immediate.** Run the F1 ablation (Limitation 2). Re-run the verifier study with task-specific negatives and an uncontaminated base-model judge (§6). Both are small and both convert an observational claim into a controlled one.

**Pipeline.** Collect execution-verified trajectories: run the agent against the live sandbox and retain a rollout only if `submit_patch` fired, the diff applied, and the reproduction test flipped red to green. This replaces the unverified synthetic majority of the training set and is the single highest-value remaining change.

**Open question.** Whether code-graph anchoring improves end-task performance remains open. The present work establishes only that the question cannot be answered by a pipeline exhibiting F1–F5 — and that a converging loss curve is not evidence that it does not.

---

## 11. Conclusion

We set out to build a graph-anchored offline coding agent and did not. What we obtained instead is a detailed account of five ways an agentic post-training pipeline can fail while reporting success: a stop token that receives no gradient, a context-alignment path that fails open, a final stage that erases the first, four serialization formats where one was needed, and a metric that counts attempts rather than outcomes.

The through-line is that none of these is visible in the training loss. A cross-entropy scalar cannot report which tokens were supervised, whether the policy was taught to stop, what the preprocessing kept, or whether the number being optimised means anything. The remedy is not a better loss but **structural assertions** — on label coverage, stop-token supervision, train/eval format identity, dataset survival, and falsifiable evaluation — run before every training job.

We report a negative result with identified mechanisms in preference to a positive result we cannot support. The verifier-alignment study in §6 stands within its caveats. The agent does not work, and we say so.

---

## References

*(To be completed — GraphForge `arXiv:2609.38923`, SWE-bench, LoRA, rejection sampling / STaR, LLM-as-a-judge reliability.)*

---

## Appendix A: Measured Chat Template and Environment Facts

Established by direct measurement before failure analysis.

**Rendered template** for a system / user / assistant-with-tool-call / tool / assistant sequence:

```
<bos><|turn>system
SYS<turn|>
<|turn>user
USR<turn|>
<|turn>model
<|tool_call>call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|><|tool_response>response:read_file{value:<|"|>OBS<|"|>}<tool_response|>THINK<turn|>
DONE<turn|>
```

**Observations.**

1. **Tool responses are folded inside the model turn**, not emitted as a separate turn. Any masking scheme that assumes one role per turn block will mis-handle this.
2. **Assistant content is emitted after the tool-call and tool-response blocks** within the same turn (`THINK` follows both).
3. **The final assistant message carries no turn-open header** — `DONE` appears directly after the preceding `<turn|>`. Masking logic keyed on turn-open markers must account for this.
4. **`<bos>` is emitted by the template**, and `add_special_tokens=True` does not duplicate it. A double-BOS hypothesis we had held is **false**.

**Special tokens.** `<bos>`, `<eos>`, `<unk>`, `<pad>`, `<mask>`, `<|turn>`, `<turn|>`, `<|tool_call>`, `<tool_call|>`, `<|tool_response>`, `<tool_response|>`, `<|tool>`, `<tool|>`, `<|"|>`, `<|think|>`, `<|channel>`, `<channel|>`, and image/audio/video markers.

**Trajectory token lengths** (n = 304): median 831, p90 1475, p95 1709, p99 2729, max 4367. At the historical 1,536-token cap, 24 of 304 (7.9%) were truncated. At 8,192, none overflow.

This distribution is what establishes F2 arithmetically: a median of 831 tokens is incompatible with a ~1,830-token preamble being present.
