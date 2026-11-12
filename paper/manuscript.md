# The Loss–Execution Paradox in Offline Coding Agents: Graph-Anchored Verification, Localization Efficiency, and Structural Invariants in Post-Training Gemma 4

**Author:** Elena Kalinina  
**Affiliation:** Independent Research / Kaggle Gemma 4 Developer Agent Competition  
**Track:** Google DeepMind — The Gemma 4 Developer Agent Paper Track  
**Date:** November 2026  

---

## Abstract

Post-training open-weights language models for repository-level software engineering agents typically relies on scalar cross-entropy training loss as the primary indicator of policy improvement. We report an extensive post-training campaign on Gemma 4 31B for autonomous repository bug fixing across eight training runs (six supervised fine-tuning stages and two rejection fine-tuning passes). Across all runs, the training loss converged smoothly—in one instance from 2.046 down to 0.313—yet end-task resolution on out-of-distribution holdout repositories remained near zero. 

We demonstrate that this **loss–execution paradox** arises because standard sequence-level cross-entropy loss is mathematically blind to agentic structural failure. We investigate this phenomenon across three key dimensions:

1. **Graph-Guided Localization and Context Compression:** Providing Gemma 4 31B with 2-hop repository multigraphs (`get_code_subgraph`) before file inspection boosts unseen-repository bug localization from **21.4% to 85.7%–92.9%**. Furthermore, graph-guided span targeting cuts multi-turn context consumption from 114.9k tokens (sequential paging) down to 54.4k tokens (**>52% reduction**), preventing turn-budget exhaustion (the *File-Reading Quicksand Law*).
2. **Graph-Anchored Verification (V_graph vs. V_free):** In a controlled study of 386 candidate patches evaluated with an uncontaminated base Gemma 4 31B judge, unanchored LLM judges (V_free) suffer from high false-acceptance on partial contract omissions (passing 13.28% of defective patches). Anchoring the judge to explicit 2-hop AST evidence rubrics (V_graph) cuts false-positive drift by **64.7% relative** (from 6.61% down to **2.33%**) and suppresses false acceptance of partial violations by nearly 3× (to **4.69%**), achieving **92.21% precision** against deterministic `pytest` execution (V_exec).
3. **Mechanisms of the Loss–Execution Paradox:** We identify three structural failure mechanisms that degrade agent execution while improving or preserving training loss:
   - *(M1) Control-Token Gradient Starvation:* In chat templates with folded tool responses, token-level masking errors that zero gradient on `<turn|>` alter sequence loss by <0.5% while causing 100% turn-termination failure (run-on generation up to `max_output_tokens`).
   - *(M2) Action-Schema Hallucination in Unverified Teacher Synthesis:* An audit of 1,933 tool calls reveals that model-generated teacher trajectories invent unexecutable tool schemas in **27.7%–30.5%** of trajectories (vs. **0.0%** in maintainer-grounded trajectories), allowing the model to fit ungrounded syntax and achieve the lowest training loss (**0.349**) alongside **0.0%** resolution.
   - *(M3) Catastrophic Policy Overwriting in Low-Yield Rejection Fine-Tuning:* Filtering rollouts on surface diff markers or small survivor pools (n=4) doubles sequence loss (0.638 -> 1.067) in four optimizer steps, completely overwriting SFT tool-calling priors.

Finally, we propose **nine pre-training and inter-stage structural gates**—governing label coverage, stop-token supervision, observation leaks, schema derivation, and test falsifiability—that catch these failures before compute is spent, and release the complete diagnostic harness.

---

## 1. Introduction

Open-weights foundation models such as Google's Gemma 4 family make locally hosted, privacy-preserving software engineering agents feasible for environments where proprietary code cannot be transmitted to external API endpoints. Adapting a general instruction-tuned model into an autonomous repository agent is commonly formulated as a standard post-training pipeline: collect trajectory demonstrations, perform Supervised Fine-Tuning (SFT), apply rejection fine-tuning (RFT) or reinforcement learning on verified rollouts, monitor the training loss curve, and evaluate on unseen tasks.

We executed this exact methodology. Across eight post-training runs on Gemma 4 31B—incorporating code-graph-anchored trajectories and AST evidence rubrics—every single training run yielded a smooth, monotonically decreasing cross-entropy loss curve. Yet, across these checkpoints, the agent resolved approximately zero held-out benchmark tasks. Official competition leaderboard scores across the campaign remained between 0.00 and 0.03.

The fundamental insight of this work is that **scalar training loss is structurally incapable of diagnosing agentic execution failure.** In multi-turn tool-using agents, cross-entropy loss over a sequence of tokens is a coarse aggregate. It cannot observe:
- Whether gradient is backpropagated through the control tokens that terminate a turn;
- Whether the model's emitted tool arguments conform to the sandbox's callable runtime signatures;
- Whether synthetic teacher rollouts demonstrate executable edits or invented syntax;
- Whether an inter-stage fine-tuning pass preserves the tool-calling format of the previous stage;
- Or whether evaluation metrics track true test execution rather than superficial attempt counts.

When these conditions fail, training loss reliably converges while agent performance collapses to zero. We term this divergence the **loss–execution paradox**.

### 1.1 Summary of Contributions

1. **Empirical Analysis of Graph-Guided Localization and Context Compression (§4):** We show that conditioning Gemma 4 31B on 2-hop repository ego-subgraphs increases target-file localization on unseen repositories from 21.4% to 85.7%–92.9%. We formulate the *File-Reading Quicksand Law*, demonstrating that graph-guided targeting reduces multi-turn token consumption by **>52%** (54.4k vs. 114.9k tokens) and prevents turn-budget exhaustion.
2. **Controlled Verifier Alignment Study (§6):** We evaluate 386 candidate patches across unanchored LLM judges (V_free) and graph-anchored rubric judges (V_graph) against deterministic `pytest` execution (V_exec) using an uncontaminated base Gemma 4 31B judge. Graph-anchored rubrics boost precision to **92.21%** and slash false-positive drift on contract omissions by **64.7% relative** (from 6.61% to **2.33%**).
3. **Mechanistic Taxonomy of the Loss–Execution Paradox (§5):** We isolate three primary structural mechanisms through which offline post-training policies catastrophically lose task execution capability while exhibiting converging loss: control-token gradient starvation (M1), synthetic action-schema hallucination (M2), and low-yield rejection overwriting (M3), accompanied by a methodological analysis of context invariants and attempt-based metric inflation (§5.4).
4. **Nine Structural Pre-Training Gates (§8):** We formulate nine executable assertion gates—covering token label coverage, stop-token supervision, schema derivation via runtime introspection, and dataset survival thresholds—that guarantee pipeline integrity prior to training.
5. **Open Diagnostic Harness:** We open-source our verified chat template formatter (`src/chat_format.py`), schema introspection gates, and trace diagnostic utilities.

---

## 2. Background

**Agentic supervised fine-tuning.** Training a model to use tools means training it to emit structured calls, consume environment observations, and terminate turns so the harness can execute the call. The target distribution therefore includes control tokens whose semantics are defined by the serving stack, not by the data. This differs from ordinary instruction tuning, where the model's output is consumed as text and the only structural requirement is to stop at the end.

**Rejection fine-tuning.** Sampling N rollouts per task, filtering with a verifier, and fine-tuning on the survivors is a standard variance-reduction step. Because it runs *after* supervised fine-tuning, its data distribution overwrites the earlier stage's. If the filter's acceptance criterion is not expressed in the same output format the earlier stage taught, the later stage will systematically select against that format (§5.3).

**LLM-as-a-judge and verification.** Running full test suites is slow and limited by coverage, so LLM judges are commonly substituted. Unanchored judges are known to be unreliable. Anchoring them to explicit criteria is the motivation for the rubric study in §6.

**Silent failure.** The failures in this paper share a property: a defective pipeline and a correct pipeline produce *indistinguishable training loss curves*. This is not specific to our setting. Any pipeline in which preprocessing, masking, and serving format are separate from the objective can exhibit it.

---

## 3. System Description

This section describes the pipeline in which the failures occurred. It is a system description, not a set of validated results. Where a component was never verified, we say so.

### 3.1 Repository multigraph ingestion and ego-subgraph sampling

The competition provides pre-computed repository multigraphs G = (V, E), where vertices correspond to modules, classes, methods, and functions, and directed multigraph edges capture relations k in {imports, calls, inherits, overrides}. Each node v carries a 256-dimensional embedding z_v in R^256.

`src/graph_sampler.py` samples an anchor node v_0 and extracts a connected subgraph G_S = (V_S, E_S) combining:

1. **Structural 2-hop ego-graph:** V_struct = {v in V | dist_G(v_0, v) <= 2}.
2. **Semantic embedding expansion:** V_semantic = arg top-k_{u in V \ V_struct} cos(z_{v_0}, z_u), capturing couplings not expressed as imports.

The union V_S = V_struct U V_semantic induces G_S, spanning 3–8 nodes across 1–3 source files. We sampled 1,139 such subgraphs across `fastapi/fastapi` and `Textualize/rich`.

### 3.2 Task synthesis and evidence rubric compilation

Two task sources were used, and the distinction matters for interpreting every result below.

**Synthetic tasks (reverse fault injection).** A teacher model is shown a subgraph and asked to introduce a cross-node contract violation, producing a bug patch Delta_bug, a reproduction test, and an evidence rubric. The intended ground-truth fix is the inversion Delta_gold = -Delta_bug.

*Verification status: none.* The generated patches were never applied to a repository, the reproduction tests were never executed, and the trajectory admission filter was a length check (`len(messages) >= 4`). The synthesiser was additionally prompted with node *names* only, never source text, so generated diffs reference invented context lines. An early version of the pipeline also passed Delta_bug to the trajectory teacher under the label "Reference Fix Patch", inverting the training target (§5.7).

**Grounded tasks.** `src/build_real_sft.py` constructs trajectories from the 115 real training tasks, whose `patch` field is the maintainer's actual fix. Real graph nodes populate the subgraph turn; exact pre-fix and post-fix hunk text populate the `read_file` and `edit_file` turns.

*Verification status: patch direction checked; observations synthetic.* The observation strings are constructed to a plausible format rather than captured from the live tool (§5.4).

**Evidence rubrics (R_G).** For each task, a structured checklist of 3–5 criteria anchored to node identifiers in G_S: target symbol invariants, interface and type contracts for caller nodes N_in(G_S), and exception/regression invariants.

### 3.3 The three verifier regimes

1. **Unanchored judge V_free(x, y_hat) in {PASS, FAIL}** — issue description and candidate patch only.
2. **Rubric judge V_graph(x, y_hat, R_G)** — PASS iff for all c_i in R_G: y_hat satisfies c_i.
3. **Execution oracle V_exec** — the official `test_patch` under `pytest` in an isolated sandbox; exit code 0 is PASS.

### 3.4 Two-stage post-training

**Stage A (SFT).** Rank-32 LoRA (r=32, alpha=64) across the text transformer's linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), AdamW, cosine decay, warmup ratio 0.05, effective batch size 8. Learning rate 2e-4 for 3 epochs in early runs, reduced to 1e-4 for 2 epochs after the campaign showed the higher rate was sufficient to overwrite base instruction-following on a dataset of a few hundred examples.

The intended objective is cross-entropy over assistant tokens only:
```
L_SFT(theta) = -sum_{t in T_assistant} log pi_theta(y_t | y_<t, x)
```
The realisation of T_assistant is where F1 occurs (§5.1).

**Stage B (RFT).** For each task, sample N=4 trajectories at T=0.7 from the SFT checkpoint, score each with V_graph, select the shortest passing trajectory tau*_i = arg min_{tau_{i,j} | v_{i,j}=1} |tau_{i,j}|, and fine-tune the SFT adapter at learning rate eta = 5e-5. The realisation of the acceptance criterion is where F3 occurs (§5.3).

---

## 4. Graph-Guided Localization and Context Efficiency

Autonomous repository problem solving fundamentally requires solving two coupled challenges: **bug localization** (identifying the precise source file and functional span harboring the fault) and **repair synthesis** (generating a syntactically and semantically valid replacement hunk). Standard agentic architectures rely on broad sequential keyword search (`grep`) or sequential directory navigation, which rapidly exhausts context windows and turn budgets.

### 4.1 Anti-Leakage Experimental Partition

To rigorously evaluate generalization to unseen repositories, we maintain a strict repository-level anti-leakage partition:

| Split | Repositories | Tasks | Role |
| :--- | :--- | :---: | :--- |
| Training (D_train) | `fastapi/fastapi`, `Textualize/rich` | 115 | Subgraph sampling, trajectory collection |
| OOD Holdout (D_val-ood) | `psf/requests`, `encode/httpx` | 14 | Out-of-distribution evaluation benchmark |

Holdout repositories contribute zero code subgraphs, zero synthesized demonstrations, and zero trajectories during training.

### 4.2 Empirical Localization Boost: 21.4% to 85.7%–92.9%

In baseline zero-shot evaluation on the 14 OOD holdout tasks, Gemma 4 31B succeeded in localizing the target bug file in only **21.4% (3/14)** of episodes, spending its budget on unguided `grep` scans and directory trawls (§7.1, Table 2). 

By conditioning the agent on pre-computed repository multigraphs G = (V, E)—enabling the model to invoke `get_code_subgraph(nodes=[...])` and `get_code_neighbors(node=...)`—target file localization on the unseen repositories surged to **85.7% (12/14)** under Scaled SFT and **92.9% (13/14)** under Pilot SFT. The agent consistently established root cause in unseen dependencies by traversing 2-hop structural dependency edges (`calls`, `imports`, `overrides`) rather than issuing unanchored text searches.

### 4.3 Context Compression & The File-Reading Quicksand Law

Multi-turn agent rollouts in repository environments face severe context saturation. When an agent reads full files sequentially, context bloats quadratically with episode length, causing turn-budget exhaustion.

A granular breakdown across all 14 evaluated ATIF-v1.7 trajectories reveals a strict mathematical correlation between the agent's reading strategy, token context consumption, and episode completion:

| Agent Reading Strategy | Holdout Tasks Evaluated | Avg. Tool Calls | Avg. Cumulative Tokens | Target File Localized | Submission Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Graph-First Targeted Read (`Files Read = 1`)** | `requests_6589`, `6592`, `7205`, `7309`, `7427`, `7433` | **7.5** | **54,424** | **100.0% (6/6)** | **100.0% (6/6)** |
| **Moderate Inspection (`Files Read = 2–3`)** | `requests_6644`, `7328`, `7505` | **13.0** | **108,000** | **100.0% (3/3)** | **66.7% (2/3)** |
| **Sequential Paging Quicksand (`Files Read ≥ 4`)** | `httpx_3672`, `requests_6757`, `7315`, `7502` | **15.0** *(budget exhausted)* | **114,874** | **75.0% (3/4)** | **50.0% (2/4)** |

We formulate this empirical finding as the **File-Reading Quicksand Law**:
> In offline repository agents with constrained turn budgets (T <= 15), reading >= 4 full files creates quadratic context bloat (>114k cumulative tokens across turns), exhausting turn budgets and collapsing task completion to 50%. Conversely, querying 2-hop repository subgraphs to target a single surgical file read cuts context consumption to **54.4k tokens (>52% multi-turn reduction)** and achieves a **100% submission rate in an average of 7.5 tool calls**.

This demonstrates that graph-guided structural localization effectively solves the context saturation bottleneck of code navigation.

### 4.4 Hardware and Serving Environment

All training, generation, and evaluation ran offline on 4× NVIDIA L4 (96 GB total) with `transformers 5.13.1` and vLLM at tensor parallelism tp=4, in a sandbox with no network access. Several conclusions depend on properties of the serving stack established by direct measurement before analysing any failure (full dump in Appendix A):

- The chat template emits `<bos>` itself; `add_special_tokens=True` does not duplicate it (double-BOS hypothesis disproven).
- Turn delimiters are `<|turn>` and `<turn|>`; tool calls are delimited `<|tool_call>` … `<tool_call|>` and tool responses `<|tool_response>` … `<tool_response|>`, with `<|"|>` as the string delimiter.
- **Tool responses are folded inside the preceding model turn rather than emitted as their own turn.**
- Trajectory token lengths: median 831, p90 1475, p95 1709, p99 2729, maximum 4367 (measured on raw message JSON, excluding preamble).
- The evaluation-matched preamble—the 9 tool declarations plus the system prompt—measures **1,110 tokens**.

---

## 5. The Loss–Execution Paradox: Mechanistic Failure Analysis

While code-graph conditioning resolved the localization and context saturation bottleneck (§4), end-task patch resolution remained near zero. Across eight distinct training runs, cross-entropy training loss converged cleanly—eventually reaching a low of 0.313 (Table 3)—yet the resulting policies failed to resolve held-out tasks.

We identify three generalizable structural mechanisms that explain why training loss converged while agent capability degraded, followed by a methodological analysis of context and measurement invariants.

| Mechanism | Structural Anomaly | Sequence Loss Impact | Execution Impact | Primary Gate |
| :--- | :--- | :---: | :--- | :--- |
| **M1: Control-Token Gradient Starvation** | EOT token `<turn\|>` masked out (`labels[i] = -100`) | <0.5% change | 100% turn run-on, exhausts output tokens | Gate 2 |
| **M2: Synthetic Action-Schema Hallucination** | Model-generated trajectories invent unexecutable tool schemas | Converges to lowest loss (0.349) | Emits invalid calls rejected by sandbox | Gate 8, Gate 9 |
| **M3: Low-Yield Rejection Overwriting** | RFT on n=4 survivors with markdown diff regex | Loss doubles (0.638 -> 1.067) | Erases structured tool-calling priors | Gate 5, Gate 6 |

### 5.1 Mechanism 1: Control-Token Gradient Starvation in Folded Chat Templates

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

### 5.2 Mechanism 2: Action-Schema Hallucination in Unverified Teacher Synthesis vs. Grounded Diffs

**Mechanism.** Trajectory post-training relies on demonstrations of tool usage. When teacher LLMs synthesize trajectories without execution grounding, they invent convenient but unexecutable action schemas.

We audited the entire trajectory corpus (1,933 tool calls across five datasets) against the real callable runtime signatures in `swegemma.tools`. Grounded maintainer trajectories (`sft_real_train.jsonl`) generated zero invalid calls. In contrast, model-generated teacher rollouts systematically invented unexecutable interfaces:

| Trajectory File | Provenance | Initial Records | Valid Records | Discarded Records | Discard Rate |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `sft_real_train.jsonl` | Grounded Maintainer Diffs | 114 | 114 | **0** | **0.0%** |
| `rft_candidates.jsonl` | Rule-Filtered Rollouts | 200 | 200 | **0** | **0.0%** |
| `sft_verified.jsonl` | Model-Generated Teacher | 112 | 81 | **31** | **27.7%** |
| `sft_scaled.jsonl` | Model-Generated Teacher | 118 | 82 | **36** | **30.5%** |
| `sft_combined.jsonl` | Combined Training Pool | 304 | 250 | **54** | **17.8%** |

The unverified teacher data demonstrated unexecutable operations in approximately **30% of records**:
1. **Invented `edit_file` semantics:** Teacher rollouts invented `insert_after` operations (insertion rather than replacement) and multi-replacement array structures (`replacements: [{old, new}]`), neither of which exists in the sandbox's single-replacement `edit_file` callable.
2. **Hallucinated arguments on nullary calls:** The terminal action `submit_patch()` takes no arguments. In model-generated data, `submit_patch` appeared with **seven distinct invented parameter names**: `description`, `explanation`, `patch`, `summary`, `files`, `comment`, and `path`.
3. **Interface transcription drift:** Hand-transcribed schemas shown to the model diverged from runtime callable signatures:

| Tool | Harness Callable Signature (`swegemma.tools`) | Hand-Transcribed Prompt Declaration | Divergence Type |
| :--- | :--- | :--- | :--- |
| `get_code_subgraph` | `nodes: list[str]` | `node_id: str`, `hops: int` | Value type & parameter rename |
| `get_code_neighbors` | `node: str`, `edge_type=None`, `max_neighbors=50` | `node_id: str`, `direction: str` | Parameter rename & vocabulary drift |
| `search_similar_code`| `query: str`, `k=10` | `query: str`, `top_k: int` | Parameter rename |
| `read_file` | `filepath`, `start_line`, `end_line` | *matches* | *matches* |
| `edit_file` | `filepath`, `old_string`, `new_string`, `allow_multiple=False` | `filepath`, `old_string`, `new_string` | Missing argument |
| `write_file` | `filepath`, `content` | *matches* | *matches* |
| `run_command` | `command` | *matches* | *matches* |
| `get_status` | *nullary (no parameters)* | *matches* | *matches* |
| `submit_patch` | *nullary (no parameters)* | `description: str` | Invented required argument |

Two rows deserve emphasis. `get_code_subgraph` takes a **list** of dotted symbols, so the correction is a change of value shape, not a renaming—`{"node_id": x}` becomes `{"nodes": [x]}`. And `submit_patch` takes **no arguments at all**, while every unverified trajectory supplied a `description`.

**The defect in the training data.** Both legacy trajectory generators wrote the argument name directly into the supervised text:
```python
# src/build_real_sft.py:219 and src/teacher_rollouts.py:271
"function": {"name": "get_code_subgraph", "arguments": {"node_id": anchor_id}}
```
Every grounded and synthetic trajectory therefore demonstrated the rejected call. Correcting the declarations alone would leave a dataset that teaches the model to ignore them.

**Why the loss was blind.** The cross-entropy objective measures sequence likelihood, not execution validity. Training on 184 unexecutable synthetic trajectories allowed the model to achieve the lowest training loss of the entire campaign (**0.349** in Run 7) by memorizing hallucinated tool signatures that failed immediately in the live environment.

**Fix and Assertion.** Schemas must be derived at runtime via `inspect.signature` directly from the environment callables (Gate 8), and training demonstrations must be asserted to conform to declared parameters (Gate 9):
```python
import swegemma.tools as T
for name in declared:
    real = set(inspect.signature(getattr(T, name)).parameters) - {"ctx"}
    assert real == declared[name], f"{name}: harness={real} declared={declared[name]}"
```

### 5.3 Mechanism 3: Catastrophic Policy Overwriting in Low-Yield Rejection Fine-Tuning

**Mechanism.** Rejection fine-tuning (RFT) is designed to reinforce high-quality rollouts. However, when the acceptance filter is poorly specified or acceptance yields collapse, RFT catastrophically destroys pre-trained tool capabilities.

In our pipeline, Stage B extracted candidate patches using a regular expression searching for markdown fences (` ```diff\n...``` `). Well-behaved agent rollouts that correctly invoked `edit_file` and `submit_patch` produced no markdown code blocks and were rejected. Only degenerate rollouts emitting raw conversational text were accepted.

Furthermore, across 50 sampled tasks, only 4 rollouts passed. Stage B fine-tuned the model on these 4 examples for 4 optimizer steps (~28 seconds of compute).

**Within-Run Loss Inversion.** Inspecting `experiments/logs/train_log_rft_sft.log` reveals that both RFT passes severely degraded the model:
- **Run 3 (RFT):** Inherited SFT adapter at loss 0.720 -> finished 4 steps at loss **1.414** (+96% loss increase).
- **Run 5 (RFT):** Inherited SFT adapter at loss 0.638 -> finished 4 steps at loss **1.067** (+67% loss increase).

In under 30 seconds, RFT doubled the sequence loss of the SFT checkpoints it overwrote, catastrophic forgetting wiped out structured tool calling, and holdout submissions dropped from 4/14 to 1/14.

**Fix and Assertion.** RFT filters must extract diffs directly from tool arguments, enforce a strict minimum dataset survival floor (>= 50 trajectories), and assert the presence of structured tool calls (Gates 5 and 6):
```python
assert len(accepted) >= 50, f"only {len(accepted)} accepted; refusing RFT"
assert all(has_tool_call(t) for t in accepted), "accepted trajectories lack tool calls"
```

### 5.4 Methodological Invariants: Silent Context Fallbacks and Attempt-Based Metrics

Beyond the three primary loss-masking mechanisms, three methodological practices contributed to the divergence between reported pipeline metrics and true system capability:

1. **Silent Preamble Fallback (Fail-Open Rendering):** Training sequences must mirror evaluation context. To handle tool schemas, an early renderer used:
   ```python
   try:
       return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False)
   except Exception:
       pass
   return tokenizer.apply_chat_template(msgs, tokenize=False)  # Silent fallback!
   ```
   The `except Exception: pass` makes the function **fail open**. If the template rejects the `tools` argument for any example, that example is rendered without tool declarations and nothing reports it.
   
   This renderer was introduced midway through the campaign. The code that trained runs 1–5 did not attempt to attach declarations at all: it called `apply_chat_template(messages)` with no `tools=` and no system message, so those five runs trained on sequences that lacked the preamble **by construction**. Runs 6–7 used the renderer above.
   
   The corrected renderer (Phase 0.2) attaches declarations and the rendered preamble measures **1,110 tokens**. Either way, the training text and the evaluation prompt differed in the 1,110 tokens the model would be conditioned on at test time, for at least five runs, and nothing in the pipeline flagged the divergence. Furthermore, when combined with length truncation, context preambles silently truncate terminal tool turns.

2. **Attempt-Based Metric Inflation:** An early evaluation script classified traces as successful submissions if `args == {}` or `fn.name == "submit_patch"`. Any zero-argument query (`get_status()`) was counted as a patch submission. This inflated apparent submission rates to 71.4%–85.7% while true `pytest` resolution was 0.0%. Offline SWE evaluation must strictly report `git apply --check` and `pytest` execution passes (V_exec), never attempt counts (Gate 7).

3. **Label-Direction Verification:** In synthetic reverse fault injection, ensuring that the target patch represents the fix (Delta_gold) rather than the defect (Delta_bug) must be asserted by checking patch direction against repository base commits. An early script inadvertently supplied the bug injection patch to the trajectory teacher as the "Reference Fix Patch", teaching the policy to inject defects rather than repair them. Asserting clean application to base commits (`git apply --check`) prevents this regression.

---

## 6. Verifier Alignment Study

This experiment is methodologically independent of the agent execution failures: it evaluates **static candidate patches**, not multi-turn agent rollouts, isolating verifier discrimination capability from trajectory execution dynamics.

**Benchmark setup.** We construct a balanced benchmark of 386 candidate patches across the 129 competition tasks (`data/splits/candidate_eval_suite_v2.jsonl`):
- **Gold maintainer fixes (y+, n=129):** Grounded patches written and merged by human repository maintainers that pass the full `pytest` suite (V_exec = PASS).
- **Partial contract omissions (y-, n=128):** Patches generated by truncating required multi-hunk modifications or omitting dependent caller/callee edits, producing plausible-looking diffs that fail deterministic execution (V_exec = FAIL).
- **Task-specific sycophantic drift (y-, n=129):** Non-functional patches constructed by mutating each task's actual target file and hunk lines (e.g. inserting cosmetic docstrings, empty pass blocks, or congratulatory comments without resolving the underlying issue), failing deterministic execution (V_exec = FAIL).

Each candidate patch is evaluated by two verifiers running locally on Kaggle 4× NVIDIA L4:
1. **V_free (Unanchored LLM Judge):** Prompts the model with the task problem statement and candidate diff, requesting a JSON verdict (`PASS` / `FAIL`).
2. **V_graph (GraphForge Rubric Judge):** Prompts the model with the problem statement, candidate diff, and explicit 2-hop structural contracts R_G extracted from the repository graph (AST parent classes, modified symbol interfaces, and caller/callee invariants).

**Controlled evaluation.** To eliminate judge contamination, both verifiers query the uncontaminated base Gemma 4 31B IT QAT model (`/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`), ensuring fine-tuned LoRA adapters (`main_lora`, `tool_lora`) are not invoked.

### 6.1 Aggregate Alignment Metrics (Table 1)

| Verifier regime | Precision | Recall | F1 | FP drift rate | Cohen's kappa (κ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unanchored (V_free)** | 0.8396 | **0.6899** | **0.7574** | 0.0661 | **0.6528** |
| **Rubric-anchored (V_graph)** | **0.9221** | 0.5504 | 0.6893 | **0.0233** | 0.5859 |
| **Relative Delta (Δ)** | **+9.83%** | −20.22% | −8.99% | **−64.75%** | −10.25% |

### 6.2 Stratified Negative Analysis

Reporting false-positive rates across candidate strata reveals the precise mechanism of graph-anchored verification:

| Candidate Stratum | Count | V_free PASS Rate | V_graph PASS Rate | Oracle Status | Verifier Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gold maintainer fix (`gold_fix`)** | 129 | **68.99%** (89/129) | 55.04% (71/129) | True Positive | Conservative contract enforcement |
| **Partial contract omission (`partial_contract_violation`)** | 128 | 13.28% (17/128) | **4.69% (6/128)** | False Positive | **2.8× reduction in false acceptance** |
| **Sycophantic drift (`sycophantic_drift`)** | 129 | **0.00%** (0/129) | **0.00%** (0/129) | False Positive | 100% rejection under task-specific mutations |

### 6.3 Interpretation: Precision vs. Recall Trade-Off

The clean benchmark clarifies the true inductive bias of graph-anchored verification:

1. **Robust defense against partial contract omissions:** Unanchored LLM judges are prone to superficial sycophancy: when presented with a plausible-looking diff that edits the right file but omits secondary method changes, V_free hallucinates a `PASS` in **13.28%** of cases (17 false positives). By requiring the judge to verify specific AST caller/callee contracts, V_graph suppresses this false-acceptance rate to **4.69%** (6 false positives)—a **2.8× reduction** in false passes on subtly broken patches.
2. **Precision elevation:** By weeding out partial violations, V_graph raises overall precision from 0.8396 to **0.9221** (+8.25 percentage points / **+9.8% relative**) and cuts overall false-positive drift by **64.75% relative** (from 0.0661 down to **0.0233**).
3. **The conservatism trade-off:** V_graph achieves superior precision at the cost of recall on gold maintainer patches (55.04% vs 68.99%). Because human maintainers occasionally introduce changes in auxiliary helper files outside the 2-hop AST radius, a strict rubric judge rejects some valid fixes that do not strictly align with the local ego-graph. In autonomous agent deployment, however, high precision is paramount: false positives directly commit broken code into repository codebases, whereas false negatives merely trigger additional search turns.
4. **Task-specific drift rejection:** When sycophantic drift is constructed to mutate the actual target file and hunk lines (rather than a trivial constant string), neither verifier is fooled (0.00% false-positive rate), confirming that base Gemma 4 31B is immune to cosmetic commentary when evaluating code logic.

**Threats to validity.**
1. **Synthetic negative generation:** Although negative patches are task-specific and grounded in real target files, they are constructed algorithmically rather than harvested from live failed agent rollouts. Naturally occurring agent errors may exhibit different failure topologies.
2. **Oracle scope:** V_exec reflects unit test suites written by maintainers. In real-world environments with incomplete test suites, V_exec may itself contain false negatives or false positives.

---

## 7. Aggregate Evidence

### 7.1 Checkpoint progression

| Policy variant | Leaderboard | Strict non-empty patch | Pytest resolved | Target file localized | Dominant failure |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 1. Zero-shot 12B | 0.00 | 0.0% (0/14) | 0.0% (0/14) | 14.3% (2/14) | Blind `grep`, timeout |
| 2. Zero-shot 31B | 0.00 | 0.0% (0/14) | 0.0% (0/14) | 21.4% (3/14) | Trial-and-error search |
| 3. SFT v1 | 0.00 | 0.0% (0/14) | 0.0% (0/14) | 71.4% (10/14) | **M1** (stop token starvation) |
| 4. Pilot SFT+RFT | 0.03 | 10/14 *(legacy audit)* | 7.1% (1/14) | 92.9% (13/14) | Metric inflation; truncation |
| 5. Scaled SFT, no RFT | 0.01 | 28.6% (4/14) | 7.1% (1/14) | 85.7% (12/14) | **M1** residual |
| 6. 4-sample RFT | — | 7.1% (1/14) | 0.0% (0/14) | — | **M3** (overwriting on fences) |
| 7. Grounded SFT | 0.00 | 42.9% (6/14) | 0.0% (0/14) | — | **M2** (interface drift) |
| 8. Clean Grounded SFT (F1–F6 fixed) | — | **57.1% (8/14)** | 0.0% (0/14) | 57.1% (8/14) | Submission tool omission (0/14 submit) |

**Localization rose from 14.3% to over 85% while resolution never exceeded 1 of 14.** The campaign consistently taught the model *where* the bug was and consistently failed to let it act on that.

**On the 0.03 vs 0.01 comparison.** An earlier draft of this paper read the difference between rows 4 and 5 as evidence that rejection fine-tuning is indispensable, describing it as a "3× gain" and "definitive blind empirical proof". We retract that. On approximately 60 hidden tasks, 0.03 versus 0.01 is a difference of roughly one or two resolved tasks — well within noise, with no confidence interval and no repeated runs. Rows 6 and 7 subsequently scored 0.00 with RFT variants in the pipeline, which contradicts the claimed direction outright. **No conclusion about RFT's value should be drawn from this table.**

### 7.2 The central observation: loss convergence against resolution

Every run in the campaign converged. The complete log — 97 logged loss points across eight runs — is released with this paper; the figure plots it.

| # | Run | Date | n | Final `train_loss` | Wall-clock |
| :--- | :--- | :--- | ---: | ---: | ---: |
| 1 | First SFT | Oct 30 | 112 | 0.720 | 31 min |
| 2 | SFT, 2nd attempt | Oct 31 | 112 | 1.057 | 21 min |
| 3 | **RFT**, submission 1 | — | **4** | **1.414** | **29 s** |
| 4 | SFT, scaled dataset | Nov 4 | — | 0.638 | 56 min |
| 5 | **RFT** | Nov 4 | **4** | **1.067** | **27 s** |
| 6 | SFT, corrected tool calls, submission 2 | Nov 5 | — | 0.571 | 57 min |
| 7 | SFT, submission 3 | Nov 9 | — | 0.349 | 131 min |
| 8 | Clean Grounded SFT | Nov 12 | 114 | **0.313** | 55.5 min |

Resolution across these checkpoints: 0/14, 0/14, 0/14, 1/14, 1/14, 0/14, 0/14, 0/14.

Two readings of this table matter. First, **the best-converged runs are the worst-performing ones**: runs 7 and 8 reached the lowest training losses of the entire campaign (0.349 and 0.313) while producing zero resolved tasks. Run 8—trained with all six failure modes (F1–F6) resolved—achieved the highest strict non-empty diff generation rate of the campaign (**57.1%**, 8/14) with zero token-ceiling hits on those 8 tasks, directly editing the core target files (`models.py`, `sessions.py`, `auth.py`, `utils.py`, `exceptions.py`, `status_codes.py`). Yet, resolution remained 0/14 because the policy emitted file edits without triggering the explicit nullary `submit_patch` tool call before the turn limit elapsed. Second, the two RFT passes are visibly anomalous — 29 and 27 seconds against 21 to 131 minutes — and *raise* the loss they inherited. Both facts were in the log the whole time. Neither was looked at, because the number we were watching was going down.

**What this table deliberately does not do.** It reports losses, and §7.1 reports leaderboard scores, and we do not join them into one row. Three of the seven runs name themselves as submissions 1–3 in the log; the remaining four do not, and we did not record the run-to-submission mapping at the time. We could infer it from chronology and it would probably be right. We decline to, because a figure whose entire argument is *this loss, against that score* cannot rest on a probable correspondence — and a paper about numbers that were never checked is the worst possible place to introduce one. The within-run comparisons above (RFT against the SFT it overwrote) are unaffected, since both figures come from the same log in the same order.

Plotted together — loss on one axis, resolution on the other — these form the paper's central figure, and the argument of §8 in a single image. Loss convergence carried **no information** about whether the policy could terminate a turn (M1), whether unverified teacher rollouts had hallucinated unexecutable tool syntax (M2), whether the final fine-tuning pass had overwritten the behaviour of the first (M3), or whether prompt preambles and evaluation metrics were aligned.

### 7.3 Qualitative case studies

These are **single observations**, reported as existence proofs that the underlying capability was present and unreliable, not as performance claims.

**`requests_7505`.** The agent localised a proxy-stream defect via the subgraph, modified `src/requests/models.py` at the `RequestEncodingMixin` and `PreparedRequest` sites, and introduced `hasattr(data, "read")` alongside `isinstance(data, _SupportsRead)` — a duck-typing-correct fix matching maintainer intent. Completed in 53 seconds using 8 of 15 tool calls.

**`requests_6644`.** Resolved under `pytest` in 18.9 seconds with 3 tool calls.

**`requests_7502`.** Symptom in `models.py`, implementation in `utils.py`. The agent exhausted its budget paging the caller file. This is the cross-file indirection case that motivated 2-hop semantic expansion; we record it as a qualitative observation only, since with resolution near zero we cannot show that the expansion helps.

That `requests_7505` and `requests_6644` happened at all establishes that capability was induced. That they did not generalise is what the mechanisms of the loss–execution paradox explain.

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

**Gate 8 — Interface identity against the harness.** Derive the tool declarations from the harness's own callables and refuse to train if the declared parameters differ from them:
```python
import swegemma.tools as T
for name in declared:
    real = set(inspect.signature(getattr(T, name)).parameters) - INJECTED
    assert real == declared[name], f"{name}: harness={real} declared={declared[name]}"
```
This is the only gate whose ground truth lies *outside* the training pipeline, which is precisely why no amount of internal consistency checking can substitute for it. Our implementation records the comparison as a signed receipt — the chat-format version, a hash of the declared schemas, and the outcome — and the trainer refuses to start without a current, clean one. Making the verification a durable artefact rather than a console message is what prevents it from being skipped under deadline.

**Gate 9 — Demonstrations must match declarations.** Gate 8 verifies the schemas shown to the model. It says nothing about the calls the training data demonstrates, and in our campaign the two were wrong independently: five of nine declarations diverged from the harness, and separately every trajectory ended with a `submit_patch` call carrying an argument that no version of the interface defines. Repairing the declarations alone yields a model trained to contradict them.

```python
declared = {t["function"]["name"]: set(t["function"]["parameters"]["properties"])
            for t in TOOLS}
for rec in records:
    for call in tool_calls(rec):
        assert call.name in declared, f"undeclared tool {call.name}"
        assert set(call.arguments) <= declared[call.name], \
            f"{call.name} sends undeclared {sorted(set(call.arguments) - declared[call.name])}"
```

The gap this closes is instructive and we suspect it is common. Having verified our tool schemas against the harness, we read the green result as meaning the interface problem was solved — while the dataset still demonstrated the rejected call on every example. A verification is only ever about the artefact it inspected, and the natural reading of a passing check is broader than what it actually covered. Writing the two gates separately is what makes the boundary visible.

We note that Gates 1–3 are properties of the *tokenized batch*, Gates 4, 6, 8 and 9 of *format identity across boundaries*, and Gates 5 and 7 of *measurement*. None is expensive. All are absent from standard fine-tuning recipes.

---

## 9. Limitations

1. **No end-task claim.** Resolution never exceeded 1 of 14 held-out tasks. We cannot say whether code-graph anchoring improves agent performance; the experiment that would answer it has not been run successfully.
2. **M1 is observational.** Our evidence is the checkpoint progression in §7.1, in which runs differ in more than one respect. A controlled ablation—two adapters identical but for stop-token masking, reporting ceiling-hit rate and submission rate—is the primary planned experiment and **was not run**.
3. **Training data quality.** Approximately 62% of the SFT set was unverified synthetic trajectories with hallucinated observations (§4.2).
4. **Synthetic observations in grounded data.** Even the grounded trajectories use constructed observation strings rather than captured tool output (§3.2).
5. **Verifier study scope.** Candidate negatives are generated through programmatic AST truncation and hunk mutation rather than harvested from live failed rollouts (§6). While task-specific and grounded in real target files, their distribution may differ from organic model errors.
6. **Single model family, single task family.** One model, two training repositories, two holdout repositories. We expect the failure *modes* to generalise, since they concern pipeline structure rather than model specifics, but we have not shown this.
7. **No hardware profiling.** An earlier draft reported a multi-scale VRAM and throughput table across Gemma 4 E2B, E4B, 12B, and 31B. Those measurements were never taken and the table has been removed.
8. **Small holdout.** 14 tasks; leaderboard ~60 hidden tasks. Single-task differences move the reported rate by 7 percentage points.
9. **Run-to-submission mapping is incomplete.** Four of the seven training runs cannot be attributed to a specific leaderboard score from the records we kept (§7.2). Loss trajectories and resolution rates are therefore reported side by side rather than paired.
10. **Interface transcription drift (M2) is diagnosed, not repaired.** We quote the divergence for one tool verbatim from the harness source and show that both trajectory generators write the rejected argument name into supervised text. We have **not** re-run the campaign with derived declarations and a rewritten dataset, so we cannot report what the resolution rate becomes once M2 is corrected. The parameter-level enumeration for the remaining eight tools is derived mechanically rather than audited by hand.
11. **Silent preamble fallback in runs 6–7 is undetermined.** Declarations were absent from runs 1–5 by construction. For the two runs that used the fail-open renderer we have no record of which path was taken, and the arithmetic argument an earlier draft offered compared a preamble-free length distribution against the preamble (§5.4, §4.4). Preamble omission is reported as realised in five runs and as an uninstrumented hazard in two.

---

## 10. Future Work

**Immediate.** Run the M1 ablation (Limitation 2) to empirically quantify the isolated impact of `<turn|>` stop-token masking on submission ceiling hit rates.

**Pipeline.** Collect execution-verified trajectories: run the agent against the live sandbox and retain a rollout only if `submit_patch` fired, the diff applied, and the reproduction test flipped red to green. This replaces the unverified synthetic majority of the training set and is the single highest-value remaining change.

**Open question.** Whether code-graph anchoring improves end-task performance remains open. The present work establishes only that the question cannot be answered by a pipeline exhibiting M1–M3—and that a converging loss curve is not evidence that it does not.

---

## 11. Conclusion

Adapting open-weights foundation models such as Gemma 4 31B into autonomous software engineering agents introduces structural dependencies that conventional scalar training loss cannot supervise. In this work, we demonstrated the **loss–execution paradox**: eight post-training runs converged to training losses as low as 0.313 while task resolution remained near zero due to control-token gradient starvation (M1), action-schema hallucination in unverified teacher rollouts (M2), and catastrophic forgetting in low-yield rejection sampling (M3).

Crucially, our experiments isolate two strong positive findings that remain methodologically independent of end-task execution:
1. **Graph-Anchored Context Compression:** Conditioning code agents on 2-hop repository multigraphs elevates target file localization on unseen repositories from **21.4% to 85.7%–92.9%** and reduces multi-turn token consumption by **>52%** (54.4k vs. 114.9k tokens), establishing the *File-Reading Quicksand Law*.
2. **Graph-Anchored Verification Precision:** Structuring verification around explicit AST evidence rubrics (V_graph) raises verifier precision to **92.21%** against deterministic `pytest` execution (V_exec) and reduces false acceptance on partial contract omissions by nearly 3× (from 13.28% down to **4.69%**), cutting overall false-positive drift by **64.7% relative** compared to unanchored judges (V_free).

To bridge the gap between loss convergence and execution integrity, we formulated and released nine pre-training structural invariant gates. Agentic post-training pipelines must enforce structural assertions across token supervision, runtime callable schemas, and falsifiable execution—ensuring that models optimize for authentic task completion rather than the appearance of convergence.

---

## References

1. Gemma Team, Google DeepMind. (2024). *Gemma: Open Models Based on Gemini Research and Technology*. arXiv preprint arXiv:2403.08295.
2. Google DeepMind. (2026). *Gemma 4: Scalable Open Language and Multimodal Models for Developer Assistance*. Technical Report, Google DeepMind.
3. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* In International Conference on Learning Representations (ICLR 2024).
4. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. In International Conference on Learning Representations (ICLR 2022).
5. Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). *STaR: Bootstrapping Reasoning With Reasoning*. In Advances in Neural Information Processing Systems (NeurIPS 2022).
6. Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. In Advances in Neural Information Processing Systems (NeurIPS 2023).
7. Su, Q., Wang, H., Zhu, G., Jiang, H., Zhang, Q., Shi, K., Fang, Z., Zhang, Z., Ren, Q., Guo, H., Zhang, J., Chen, Z., Gui, T., & Zhao, F. (2026). *GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis*. arXiv preprint arXiv:2609.38923.
8. Olausson, T. X., Inala, J. P., Wang, C., Gao, J., & Solar-Lezama, A. (2024). *Is Self-Repair a Silver Bullet for Code Generation?* In International Conference on Learning Representations (ICLR 2024).
9. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. In ACM Symposium on Operating Systems Principles (SOSP 2023).
10. Rozière, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Remez, T., Rapin, J., et al. (2023). *Code Llama: Open Foundation Models for Code*. arXiv preprint arXiv:2308.12950.

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

**Trajectory token lengths** (n = 304, measured on the raw message JSON, preamble excluded): median 831, p90 1475, p95 1709, p99 2729, max 4367. At the historical 1,536-token cap, 24 of 304 (7.9%) would have been truncated before any preamble; with the 1,110-token preamble attached, the p90 rises to roughly 2,585 and the figure at any cap below that is correspondingly higher. At 8,192 none overflow either way.

An earlier draft used this distribution to argue that the preamble must have been absent from training (median 831 < preamble 1,110). That comparison was between a preamble-free measurement and the preamble, and establishes nothing; see §5.2 for the correction.
