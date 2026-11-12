# The Loss–Execution Paradox in Offline Coding Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Competition: Gemma 4 Developer Agent](https://img.shields.io/badge/Kaggle-Gemma_4_Developer_Agent-orange.svg)](https://www.kaggle.com/competitions/gemma-4-developer-agent)

This repository accompanies the research paper:  
**"The Loss–Execution Paradox in Offline Coding Agents: Graph-Anchored Verification, Localization Efficiency, and Structural Invariants in Post-Training Gemma 4"** ([`paper/manuscript.md`](paper/manuscript.md)).

---

## Executive Summary

Adapting open-weights foundation models such as Google's Gemma 4 31B into autonomous software engineering agents typically relies on scalar sequence-level cross-entropy training loss as the primary indicator of policy improvement.

Across an extensive campaign of eight post-training runs (six supervised fine-tuning stages and two rejection fine-tuning passes), cross-entropy training loss converged smoothly—in one instance from 2.046 down to 0.313. Yet, end-task resolution on out-of-distribution holdout repositories remained near zero (official competition leaderboard scores between 0.00 and 0.03).

This divergence—the **loss–execution paradox**—occurs because scalar cross-entropy loss is mathematically blind to agentic structural failure. Cross-entropy measures sequence likelihood over static tokens; it cannot observe whether control tokens receive gradient, whether tool calls match executable runtime signatures, or whether inter-stage tuning preserves tool-calling format.

---

## Two Core Positive Findings

While end-task resolution remained near zero, our experiments isolate two strong empirical findings that are methodologically independent of rollout task execution:

### 1. Graph-Guided Localization & The File-Reading Quicksand Law (§4)
- **Localization surge:** Conditioning Gemma 4 31B on 2-hop repository multigraphs (`get_code_subgraph`, `get_code_neighbors`) elevates target-file localization on unseen repositories from **21.4% (3/14)** under unanchored search to **85.7%–92.9% (12–13/14)**.
- **Context compression:** Querying 2-hop code subgraphs enables surgical 1-file inspection, cutting multi-turn token consumption by **>52%** (54.4k vs. 114.9k tokens) and achieving a 100% submission rate in an average of 7.5 tool calls.
- **The Quicksand Law:** Reading >= 4 full files creates quadratic context bloat (>114k cumulative tokens across turns), exhausting the 15-turn budget and halving task completion.

### 2. Graph-Anchored Verification Precision (§6)
In a controlled study of 386 candidate patches evaluated with an uncontaminated base Gemma 4 31B judge against deterministic `pytest` execution (`V_exec`):

| Verifier regime | Precision | Recall | F1 | False-Positive Drift Rate | Cohen's kappa (κ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unanchored (V_free)** | 0.8396 | **0.6899** | **0.7574** | 0.0661 | **0.6528** |
| **Rubric-anchored (V_graph)** | **0.9221** | 0.5504 | 0.6893 | **0.0233** | 0.5859 |
| **Relative Delta (Δ)** | **+9.83%** | −20.22% | −8.99% | **−64.75%** | −10.25% |

- **Suppression of partial omissions:** Unanchored LLM judges suffer from superficial sycophancy, falsely passing **13.28%** (17/128) of defective patches that omit dependent caller/callee edits. Structuring verification around explicit 2-hop AST evidence rubrics (`V_graph`) suppresses this false acceptance to **4.69%** (6/128)—a **2.8× reduction**.
- **Precision elevation:** `V_graph` elevates overall precision to **92.21%** and cuts false-positive drift by **64.75% relative** (0.0661 down to **0.0233**).

---

## Mechanisms of the Loss–Execution Paradox (§5)

We isolate three structural failure mechanisms that degrade agent execution while improving or preserving training loss:

| Mechanism | Structural Anomaly | Sequence Loss Impact | Execution Impact | Primary Gate |
| :--- | :--- | :---: | :--- | :---: |
| **M1: Control-Token Gradient Starvation** | EOT token `<turn\|>` masked out (`labels[i] = -100`) | <0.5% change | 100% turn run-on, exhausts output tokens | Gate 2 |
| **M2: Synthetic Action-Schema Hallucination** | Model-generated trajectories invent unexecutable tool schemas | Converges to lowest loss (0.349) | Emits invalid calls rejected by sandbox | Gate 8, Gate 9 |
| **M3: Low-Yield Rejection Overwriting** | RFT on n=4 survivors with markdown diff regex | Loss doubles (0.638 -> 1.067) | Erases structured tool-calling priors | Gate 5, Gate 6 |

---

## Eight-Run Checkpoint Progression (§7)

| # | Run | Date | n | Final `train_loss` | Strict Patch Rate | Pytest Resolved | Dominant Failure Mode |
| :- | :--- | :--- | ---: | ---: | :---: | :---: | :--- |
| 1 | First SFT | Oct 30 | 112 | 0.720 | 0.0% (0/14) | 0.0% (0/14) | Blind search, timeout |
| 2 | SFT, 2nd attempt | Oct 31 | 112 | 1.057 | 0.0% (0/14) | 0.0% (0/14) | Trial-and-error search |
| 3 | **RFT**, submission 1 | — | **4** | **1.414** | 0.0% (0/14) | 0.0% (0/14) | M3 (loss inversion, tool erasure) |
| 4 | SFT, scaled dataset | Nov 4 | — | 0.638 | 28.6% (4/14) | 7.1% (1/14) | M1 residual, metric inflation |
| 5 | **RFT** | Nov 4 | **4** | **1.067** | 7.1% (1/14) | 0.0% (0/14) | M3 (overwriting on fences) |
| 6 | SFT, corrected calls | Nov 5 | — | 0.571 | 0.0% (0/14) | 0.0% (0/14) | M2 (interface drift) |
| 7 | SFT, submission 3 | Nov 9 | — | 0.349 | 42.9% (6/14) | 0.0% (0/14) | M2 (memorized hallucinated schema) |
| 8 | **Clean Grounded SFT** | Nov 12 | 114 | **0.313** | **57.1% (8/14)** | 0.0% (0/14) | Submission tool omission (0/14 submit) |

Across all eight runs, the best-converged runs (0.349 and 0.313) exhibited zero task resolution. Run 8—trained with M1–M3 resolved on grounded maintainer fixes—achieved the highest strict non-empty diff generation rate of the campaign (**57.1%**, 8/14) targeting core bug modules (`models.py`, `sessions.py`, `auth.py`, `utils.py`, `exceptions.py`, `status_codes.py`) with zero token-ceiling hits, yet resolution remained gated behind terminal `submit_patch` tool invocation.

---

## Nine Structural Pre-Training Gates (§8)

Agentic post-training pipelines require assertions on structure, because scalar loss observes none of it. We provide runnable, GPU-free test gates in `tests/`:

```bash
# Verify Gates 1-3: Label coverage, stop-token supervision (<turn|>), no observation leaks
python3 tests/test_chat_format_masking.py

# Verify Gate 8: Runtime callable interface identity against competition harness
python3 tests/test_tool_schema_gate.py
```

Both run offline in under a second.

- **Gate 1 (Label Coverage):** Asserts supervised token fraction falls strictly within expected range (3%–45%) and decodes trainable tokens.
- **Gate 2 (Stop-Token Supervision):** Asserts the turn-terminating token `<turn|>` carries gradient (`labels[i] != -100`).
- **Gate 3 (No Observation Leakage):** Asserts folded tool observation blocks are completely masked out.
- **Gate 4 (Prompt Identity):** Asserts train/eval prompt equivalence to prevent silent preamble drops.
- **Gate 5 (Dataset Survival Floor):** Asserts >= 50 trajectories survive preprocessing before fine-tuning starts.
- **Gate 6 (Inter-Stage Format Identity):** Asserts accepted rollouts contain valid tool calls before RFT tuning.
- **Gate 7 (Falsifiable Evaluation):** Rejects attempt-based proxy metrics; strictly requires `git apply --check` and `pytest` execution.
- **Gate 8 (Runtime Callable Introspection):** Derives declarations via `inspect.signature` from live harness callables (`swegemma.tools`).
- **Gate 9 (Demonstration Conformance):** Verifies training trajectories demonstrate only parameters present in declared schemas.

---

## Repository Structure

```text
repographforge-gemma4/
├── paper/
│   └── manuscript.md             # The complete research paper
├── src/
│   ├── chat_format.py            # Canonical chat formatter, masking, & Gate 8 receipt
│   ├── build_real_sft.py         # Grounded trajectories from real maintainer fixes
│   ├── teacher_rollouts.py       # Teacher rollout synthesizer
│   ├── rft_sampler.py            # Rejection fine-tuning sampler & rubric evaluator
│   ├── graph_sampler.py          # 2-hop ego-graph + semantic embedding expansion
│   ├── task_synthesizer.py       # Reverse fault injection task generator
│   ├── rubric_generator.py       # Evidence rubric (R_G) compiler
│   └── eval_verifier_alignment.py# Table 1 verifier alignment evaluator
├── tests/
│   ├── test_chat_format_masking.py # Offline tests for Gates 1-3
│   └── test_tool_schema_gate.py    # Offline tests for Gate 8
└── experiments/logs/             # Evaluation logs backing reported metrics
    ├── verifier_alignment_table1_v2.json  # Final Table 1 benchmark (n=386)
    ├── holdout_trace_forensics.txt        # Cell 12B trace extraction
    └── train_log_rft_sft.log              # Raw training logs for 8 runs
```

---

## References

1. Gemma Team, Google DeepMind. (2024). *Gemma: Open Models Based on Gemini Research and Technology*. arXiv preprint arXiv:2403.08295.
2. Google DeepMind. (2026). *Gemma 4: Scalable Open Language and Multimodal Models for Developer Assistance*. Technical Report, Google DeepMind.
3. Jimenez, C. E. et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* In ICLR 2024.
4. Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. In ICLR 2022.
5. Zelikman, E. et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning*. In NeurIPS 2022.
6. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. In NeurIPS 2023.
7. Su, Q., Wang, H., Zhu, G., Jiang, H., Zhang, Q., Shi, K., Fang, Z., Zhang, Z., Ren, Q., Guo, H., Zhang, J., Chen, Z., Gui, T., & Zhao, F. (2026). *GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis*. arXiv preprint arXiv:2609.38923.
8. Olausson, T. X. et al. (2024). *Is Self-Repair a Silver Bullet for Code Generation?* In ICLR 2024.
9. Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. In SOSP 2023.
10. Rozière, B. et al. (2023). *Code Llama: Open Foundation Models for Code*. arXiv preprint arXiv:2308.12950.
