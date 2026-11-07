# RepoGraphForge — Remediation Plan

Companion to `DIAGNOSIS.md`. Nothing here has been applied yet — this is for review.

**Hard constraint:** paper-track deadline is **Nov 12**; today is **Nov 7**. Five days.
The plan is therefore split into a **critical path** (Phases 0–4, must land) and
**stretch work** (Phases 5–6, only if the critical path finishes early). Phase 7 (paper
corrections) is mandatory regardless of how far the code gets.

**GPU budget:** Phases 0–3 are 100% CPU. Only Phase 4 costs quota (~1 GPU-hour per
retrain+eval cycle). Budget for **three** cycles, not one.

---

## Open questions — please answer before I start

These change what I do, and I can't resolve them from the repo:

1. **Which `agent.yaml` does the grader actually read?** `submission/agent.yaml` uses
   `model.adapter_path` / `agent.max_turns` / `agent.system_prompt`. Cell 8 of the master
   notebook writes a completely different schema (`name` / `model` / `adapter` /
   `instruction` / `tools`) into `sample_submission/agent.yaml`. One is dead config.
2. **What is the 44-task holdout?** The repo splits on `requests`+`httpx` and gets 14
   (`data_preparation` Cell 2, `paper/manuscript.md:293`). Your 44-task run isn't in the
   repo. Is it a new split, or the public leaderboard subset? I need the task list to
   build an honest eval.
3. **How much GPU quota is left?** The log says ~26h of 30h, but that was several
   milestones ago. This decides whether Phase 5 is reachable at all.
4. **Do you still have the SFT v2 adapter** (the one that produced the working
   `requests_7505` patch), or has it been overwritten in place by the RFT pass?
   `train_rft.py` writes `OUTPUT_ADAPTER_DIR == SFT_ADAPTER_DIR` — same path. If there's
   no copy, that checkpoint is gone, and Phase 1's "did this help?" baseline gets weaker.

---

## Phase 0 — Establish ground truth (CPU, ~1 h)

Several fixes depend on facts I could not verify locally (no tokenizer, no ADK here).
Do not write any fix before this phase answers them. Output goes in
`experiments/logs/phase0_facts.md`.

### 0.1 Dump the real chat template behaviour
In a Kaggle CPU notebook with the Gemma 4 tokenizer loaded:

```python
msgs = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "USR"},
    {"role": "assistant", "content": "THINK",
     "tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
    {"role": "tool", "name": "read_file", "tool_call_id": "call_0", "content": "OBS"},
    {"role": "assistant", "content": "DONE"},
]
rendered = tokenizer.apply_chat_template(msgs, tokenize=False)
print(repr(rendered))
print(tokenizer.tokenize(rendered)[:40])
print("BOS in template:", rendered.startswith(tokenizer.bos_token))
print("add_special re-adds BOS:",
      tokenizer("x", add_special_tokens=True)["input_ids"][0] == tokenizer.bos_token_id)
print("special tokens:", tokenizer.all_special_tokens)
print("additional:", tokenizer.additional_special_tokens)
```

**Answers:** the real turn-open / turn-close token strings (`<|turn>` vs
`<start_of_turn>` — the repo uses both and at most one is right), whether `arguments`
should be a dict or a JSON string, whether the `tool` role is supported at all, and
whether B4 (double-BOS) is real.

### 0.2 Confirm the tool-call format the serving stack parses
Start vLLM on the **base** model (no adapter), send one chat request with `tools=[…]`
declared, and dump the raw `message` object:

```python
print(json.dumps(resp["choices"][0]["message"], indent=2))
```

**Answers:** whether the harness expects native `tool_calls` in the response object or a
text-embedded format, and which of the repo's four serialisations (C1) is correct.
Everything in Phase 3 keys off this.

### 0.3 Measure trajectory token lengths

```python
lens = [len(tokenizer(render(ex))["input_ids"]) for ex in trajectories]
print(np.percentile(lens, [50, 90, 95, 99]), max(lens))
print("truncated at 1536:", sum(l > 1536 for l in lens), "/", len(lens))
```

**Answers:** exactly how much of the training set B2 was silently destroying. This
number goes in the paper as a finding.

### 0.4 Snapshot the current adapter
`cp -r submission/adapters/main_lora experiments/adapters/rft_v1_baseline` before
anything else touches it. Non-negotiable — it's the only record of the current
behaviour, and it's the thing new runs get compared against.

**Gate:** do not proceed until 0.1–0.3 are written down. If 0.1 shows the masking tokens
never existed (i.e. `convert_tokens_to_ids` was returning unk), then every reported loss
curve in `PROGRESS_LOG.md` is suspect and the manuscript's training section needs a
bigger rewrite than Phase 7 assumes.

---

## Phase 1 — Fix the training loop (CPU, ~2 h)

All edits in `notebooks/repographforge_clean_master.md` Cell 6 and `src/train_qlora.py`.
Keep the two in sync this time, or delete one. **I'd delete `src/train_qlora.py`** — it's
a stale fork of the notebook cell (it still has the `json` NameError, B5) and having two
divergent trainers is how C1 happened.

### 1.1 Replace the masking loop (fixes B1)
Drop the hand-rolled token scan. Render each assistant turn separately so the span
boundaries are known exactly, rather than inferred:

```python
def build_example(msgs, tokenizer, max_len):
    input_ids, labels = [], []
    for i, m in enumerate(msgs):
        prefix = tokenizer.apply_chat_template(msgs[:i],   tokenize=True, add_special_tokens=(i == 0))
        upto   = tokenizer.apply_chat_template(msgs[:i+1], tokenize=True, add_special_tokens=(i == 0))
        span = upto[len(prefix):]
        input_ids += span
        trainable = m["role"] in ("assistant", "model")
        labels += span if trainable else [-100] * len(span)
    return {"input_ids": input_ids[:max_len], "labels": labels[:max_len]}
```

The assistant span produced this way **includes its terminating end-of-turn token**,
which is the whole point — that token must carry gradient.

### 1.2 Add the assertions that would have caught this
Run on 5 random examples at the top of training, print to stdout, and `assert`:

```python
ids, lab = ex["input_ids"], ex["labels"]
assert any(l != -100 for l in lab),            "all labels masked"
assert any(l == -100 for l in lab),            "nothing masked — user turns are trainable"
eot = tokenizer.convert_tokens_to_ids(EOT)     # from Phase 0.1
assert eot in [l for l in lab if l != -100],   "end-of-turn token is not trainable"
print(tokenizer.decode([l for l in lab if l != -100]))
```

**Acceptance:** that `print` shows assistant turns only, each ending in the end-of-turn
marker, with no user text and no tool observations. Read it with your own eyes on three
samples. This is the single check that distinguishes a working run from the last four.

### 1.3 Length policy (fixes B2)
`MAX_LEN = 8192`. **Drop** examples that still overflow instead of truncating, and log
the count. Truncation silently deletes the label; dropping is visible.

### 1.4 `add_special_tokens` hygiene (fixes B4)
Per Phase 0.1. If the template emits BOS, tokenize with `add_special_tokens=False`
everywhere. Assert `input_ids.count(bos_id) == 1`.

### 1.5 Delete `src/train_qlora.py` (fixes B5)
Or, if you want it kept for the paper's artifact list, add `import json` and make it a
thin wrapper that imports the notebook's `build_example`. One tokenization path, not two.

**Gate:** Phase 1 is done when the 1.2 `print` output is correct on three samples. No GPU
spent yet.

---

## Phase 2 — Fix the inverted label (CPU, ~1 h)

### 2.1 Stop training on the bug-introducing diff (fixes A1)
`src/teacher_rollouts.py:81,91` and `data_preparation` Cell 8 both pass `bug_patch` under
the label `### Reference Fix Patch`. Two options — **I recommend (b)**:

**(a) Invert the diff.** Swap `+`/`-` on hunk bodies and swap the `---`/`+++` headers.
Cheap, but it inherits A2 (the diff was written against code the synthesiser never saw),
so the "fix" still references invented context lines.

**(b) Use the real tasks instead.** `data/splits/train_tasks.jsonl` holds **115 real
tasks** whose `patch` field is the actual gold fix against actual code. They are currently
unused for SFT. Build the trajectory label from `task["patch"]`, keep the synthetic tasks
only as extra *problem statements* if you want volume.

Either way: **rename the field in the prompt to `fix_patch`** and add
`assert "fix" in field_name` at the call site, so this specific inversion can't recur.

### 2.2 Sanity-check the direction
For 10 examples, apply the label diff to the repo at the task's base commit with
`git apply --check`. If a "fix" patch applies cleanly to code that is already fixed, the
direction is still wrong. Print pass/fail per example.

**Gate:** 10/10 label diffs apply in the correct direction.

---

## Phase 3 — Fix the train/eval contract (CPU, ~3 h)

### 3.1 One tool-call serialiser (fixes C1)
Write `src/chat_format.py` with a single `render(messages, tools) -> str` built on the
Phase 0.2 finding. Make `teacher_rollouts.py`, `rft_sampler.py`, the SFT cell and the RFT
cell all import it. Delete the four bespoke format strings
(`teacher_rollouts.py:158`, `rft_sampler.py:38-42`, the Cell 10 regex, the
`train_qlora.py` string-vs-dict variant).

### 3.2 Train with the eval-time context shape (fixes C2)
Every training example must be rendered with the **same** system prompt and the **same**
`tools=` schema list that the eval harness sends. Load both from the submission config so
they cannot drift:

```python
sys_prompt = (AGENT_DIR / "prompts/system.md").read_text()
tools      = load_tool_schemas(AGENT_DIR / "agent.yaml")
rendered   = render([{"role": "system", "content": sys_prompt}] + msgs, tools=tools)
```

**Acceptance:** diff the first 300 characters of a rendered training example against the
first 300 characters of a real eval-time prompt captured from the vLLM request log. They
should match to the byte up to the task text.

### 3.3 Raise the output cap (fixes C3)
`configs/sampling.yaml` → `max_output_tokens: 4096`. Then reconcile with
`submission/agent.yaml` (`max_tokens: 4096`) once question 1 above is answered, and
**delete whichever file is dead**.

### 3.4 Replace the anti-hesitation prohibitions with a budget (fixes C4)
Current rules forbid reading a second file and mandate `edit_file` on turn 3 — which
forces a blind patch and forbids the cross-file hop your own §RQ2 says `requests_7502`
needs. Proposed replacement:

```
You have {N} tool calls. Spend them like this:
- Start with get_code_subgraph on the symbol named in the traceback.
- Read at most 3 files. Prefer the callee the subgraph points at over scanning callers.
- Apply the fix with edit_file.
- If you have calls left, run the relevant test with run_command before submitting.
- Always finish with submit_patch. A submitted imperfect patch scores higher than none.
```

This is a behavioural change with real risk — it's the one item here I'd A/B rather than
assume. See Phase 4.2.

### 3.5 Fix the audit metric (fixes C5)
Cell 15: delete the `args == {}` clause from the submit detection. It counts any
zero-argument tool call (e.g. `get_status()`) as a patch submission, so the reported
71.4% / 85.7% submission rates in the log and manuscript are inflated by an unknown
amount. Recompute the historical traces with the corrected metric before quoting any
number in the paper.

**Gate:** 3.2's byte-level prompt match passes.

---

## Phase 4 — Retrain and evaluate (GPU, ~1 h per cycle, budget 3 cycles)

### 4.1 SFT only — no RFT (fixes B3, first half)
Delete Cells 10 and 11 from the run for now. The RFT stage trains on 4 markdown-fence
examples with no masking, judged by the policy itself, and it runs *last* — it is
overwriting whatever SFT learns with "answer in prose." Until SFT is healthy it is
strictly net-negative.

Train: `MAX_LEN=8192`, r=32, alpha=64, **lr=1e-4** (down from 2e-4 — at 112 samples,
2e-4 over 3 epochs is enough to overwrite base instruction-following), 2 epochs, cosine.

### 4.2 Evaluate three arms on the same holdout
Run the same task list (question 2) against:
- **A:** `experiments/adapters/rft_v1_baseline` + current prompt — the control
- **B:** new SFT adapter + current anti-hesitation prompt
- **C:** new SFT adapter + Phase 3.4 budget prompt

B-vs-A isolates the training fixes; C-vs-B isolates the prompt change (3.4). Without C
you won't know which of the two helped, and you'll carry the wrong one into the paper.

Report per arm: patch-submission rate (with the **corrected** 3.5 metric), diff-applies
rate, tests-pass rate, mean tool calls, mean output tokens, and **count of responses that
hit `max_output_tokens`** — that last one is the direct read-out on whether B1 is fixed.

### 4.3 Decision gate
- **Tests-pass > 0 on arm B or C** → the critical path worked. Package that arm, submit,
  write Phase 7, and treat Phases 5–6 as stretch.
- **Submissions up but tests still fail** → A1 was real but A2 (invented code) is now the
  binding constraint. Go to Phase 5.
- **No improvement** → stop and re-read the 1.2 `print` output. Something in Phase 0's
  assumptions was wrong; do not spend the third cycle blind.

---

## Phase 5 — Grounded, verified trajectories (GPU-heavy, STRETCH)

Only if Phase 4.3 says so and quota allows. This is the real fix for A2/A3 and it is
where a genuine result would come from, but it will not fit before Nov 12 alongside
everything else.

### 5.1 Put real code in the synthesiser prompt (A2)
Read the actual source span for each subgraph node (`file` + line range from the node
metadata) and include it. Then `git apply --check` every generated `bug_patch` and discard
what doesn't apply. Expect to lose most of the 350 — that's the point.

### 5.2 Collect trajectories by running the agent, not by writing fiction (A3)
Replace `generate_teacher_trajectory` (which asks the model to *compose* a 4-turn
transcript, hallucinated tool outputs and all) with a real rollout loop against the
sandbox. Accept a trajectory only if: `submit_patch` was actually called **and** the diff
applies **and** the repro test flips red→green. Target ≥200 accepted from 115 real tasks
at N=8.

Also persist `reproduction_test` — Cell 7 generates it and then drops it on the floor, so
nothing can be verified downstream even in principle.

---

## Phase 6 — Rebuild RFT properly (STRETCH, after 5)

Only once SFT is healthy and ≥50 trajectories can be accepted. Required changes:
rollouts carry real tool calls (not ` ```diff ` fences); masking per Phase 1.1;
judge is the **base** model plus actual test execution, never `main_lora` grading itself;
hard floor of 50 accepted trajectories before the pass runs at all.

---

## Phase 7 — Correct the manuscript (CPU, ~2 h, MANDATORY)

Independent of how far the code gets, these claims are not supported by the code as
written and should not go to the paper track uncorrected:

1. **"Verified" / "pristine" trajectories.** The only filter is `len(msgs) >= 4`
   (`teacher_rollouts.py:131`). Nothing is executed anywhere. Rename the artifacts and the
   prose, or implement Phase 5 and earn the word.
2. **Table 1's negative set is partly a constant.** One of the two negative classes is the
   fixed string `"diff --git a/fix.py b/fix.py\n+ # Handled edge case defensively\n+ pass"`
   (`data_preparation` Cell 4) — identical across all 129 tasks. A third of the benchmark
   is trivially separable, which inflates both verifiers. Either re-run Table 1 with
   realistic negatives (e.g. patches from failed rollouts) or report the result
   stratified by negative class so the reader can see it.
3. **The `V_graph` RFT filter result (8% vs 4%).** The judge is `"model": "main_lora"`,
   the policy being trained. Self-judging needs to be stated, or re-run with the base
   model as judge.
4. **Submission rates (71.4%, 85.7%).** Computed with the C5-buggy metric. Recompute.
5. **The SFT v1 "text-tool serialization trap" finding.** If Phase 0.1 shows the masking
   tokens never resolved, that failure mode was caused by B1, not by the absence of
   structural token wrapping — the stated conclusion would be wrong. Re-examine after
   Phase 0.

There's a real paper in the corrected version — "end-of-turn masking silently disables
agent termination, and here's the ablation" is a more useful contribution than the current
framing, and you'd have the A/B data from 4.2 to support it.

---

## Critical path summary

| Phase | Cost | Day | Blocking? |
|---|---|---|---|
| 0 Ground truth | CPU ~1 h | Nov 7 | yes — everything depends on it |
| 1 Training loop (B1/B2/B4/B5) | CPU ~2 h | Nov 7–8 | yes |
| 2 Inverted label (A1) | CPU ~1 h | Nov 8 | yes |
| 3 Train/eval contract (C1–C5) | CPU ~3 h | Nov 8–9 | yes |
| 4 Retrain + 3-arm eval | GPU ~3 h | Nov 9–10 | yes |
| 5 Grounded data (A2/A3) | GPU heavy | Nov 10–11 | stretch |
| 6 RFT rebuild | GPU | Nov 11 | stretch |
| 7 Manuscript corrections | CPU ~2 h | Nov 11–12 | yes |

Phases 0–3 are ~7 hours of CPU work and cost no quota. If only one thing gets done,
make it **Phase 1.2** — print the unmasked training text and read it.
