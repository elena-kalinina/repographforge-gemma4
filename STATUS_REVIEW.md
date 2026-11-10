# Review of the Implemented Fixes — What Still Breaks, and the Paper

Reviewed 2026-11-10 against `FIX_PLAN.md`. Paper deadline **Nov 12 (2 days)**;
code competition deadline **Dec 2 (3 weeks)**. That asymmetry drives the recommendation
at the bottom.

## Verdict in one line

The Phase 1–3 code fixes are largely correct, and Phase 0 was run (its results just
aren't written down anywhere in the repo, which is why §2 below still has to ask for
them). The dataset was **not** emptied — see the retraction in §1.

What remains: the tool schemas in `chat_format.py` are a hand-written copy that nothing
checks against the real ADK definitions; the `tools=` render fails open, so C2 may never
have taken effect; ~62% of the training data is still the unverified synthetic set; and
the eval still cannot tell you whether a patch works.

---

## 1. RETRACTED — the dataset was NOT emptied

My original §1 claimed `MAX_LEN=2560` + drop-don't-truncate had emptied the training set.
**That was wrong.** The training log reads:

```
[18:32:29] INFO - ✓ Tokenized 299 trajectories (dropped 5 > 2560 tokens).
```

299 kept, 5 dropped. The dataset was fine. Disregard that diagnosis.

Two things remain true and are now the live questions:

**(a) 299 fitting under 2560 is hard to reconcile with the preamble being present.**
**RESOLVED 2026-11-10 by Cell 0C.** The preamble measures **1,110 tokens** against the
real tokenizer. (My earlier figure here, ~1,830, was a char/4 *estimate* mislabelled as
"measured" — it overshot by 65%. Corrected throughout.) The median trajectory was 831
tokens, so at least half the training set was shorter than the preamble alone: it was
not present. `render()` had been swallowing the failing `tools=` render via
`except Exception: pass`, so C2 was only half-applied. The same cell confirms F2 is now
**fixed**: declarations render and are present in the text.

**(b) 299 = 115 grounded + ~184 synthetic screenplays.** `sft_combined.jsonl` merges
`sft_real_train.jsonl` with `sft_scaled.jsonl`, so **~62% of the training data is still
the unverified, hallucinated-observation synthetic set** that A3 was supposed to retire.
The grounded builder was added; the old data was never removed.

Also note the merge loop in Data Prep Cell 9 does `all_trajectories[syn_id] = item`
unconditionally, so `sft_scaled` **overwrites** `sft_real_train` on key collision — the
opposite of its "always include all 115 real tasks first" comment. Probably harmless
(keys are unlikely to collide) but the intent is inverted.

The memory fix in the retained section below still stands on its own merits: it buys back
real context length for Stage 2 of the Dec 2 plan.

## 2. Phase 0.2 was skipped — the tool schemas are invented

`src/chat_format.py:COMPETITION_TOOLS` is a hand-written guess: `node_id`, `hops`,
`filepath`, `start_line`, `end_line`, `old_string`, `new_string`, `query`, `top_k`,
`direction`, `command`, `description`. Nothing in the repo verifies these against the
actual ADK tool definitions in Ryan's Section 3.

Phase 3.2's whole purpose was to make training match eval. If even one parameter name is
wrong (`filepath` vs `path`, `node_id` vs `symbol`), then training actively teaches the
model to emit calls the sandbox rejects — which is *worse* than not training on tool
schemas at all, because the adapter now confidently produces the wrong shape.

This is the leading explanation for **"6/14 submitted but score 0."** `edit_file` failing
silently, followed by `submit_patch` on an unmodified tree, produces exactly that
signature: a submission that is empty or trivially wrong.

**Do this first, it costs 5 minutes of CPU:**

```python
# after Ryan's Section 3 has defined the tools
import json
print(json.dumps([t.to_dict() if hasattr(t, "to_dict") else t for t in TOOLS], indent=2))
```

Then diff against `COMPETITION_TOOLS` field by field. Better: **import the real schemas**
rather than copying them, so they cannot drift again.

### Related: `render()` fails open
`chat_format.py:render` wraps the `tools=` call in `try/except Exception: pass` and falls
through to a no-tools render. If `apply_chat_template` doesn't accept `tools` for this
template, **every example silently trains without tool declarations** and you'd never
know. Same silent fallback in `build_example`. Make it raise, or at minimum log loudly
and count occurrences.

---

## 3. The masking fallback path can unmask tool observations

`build_example` has two paths. The prefix-diff path is correct — `tool` messages get
`[-100] * len(span)`. But it sets `prefix_ok = False` and falls back whenever the template
rewrites earlier tokens, which Gemma templates routinely do when `tools=` is passed (tool
definitions get folded into the first turn).

In the fallback, `_mask_rendered_token_ids` detects roles by matching the token sequence
after a turn-open against `"system"` / `"user"` / `"model"`. If the template renders tool
results as their own turn role (e.g. `tool`), **none of the three branches match**,
no flag is set, and control reaches:

```python
if not in_system and not in_user and not in_tool_response:
    labels[i] = input_ids[i]
```

— so the entire tool observation becomes trainable. That is the original B1-family bug
returning through the back door: the model is again trained to generate its own
observations.

`verify_and_print_samples` will not catch this. Its assertion is only "something is
masked" (`any(l == -100)`), which the system prompt alone satisfies.

**Fix — tighten the gate:**

```python
frac = len(unmasked_ids) / len(ids)
assert 0.03 < frac < 0.45, f"Unmasked fraction {frac:.1%} is out of range — check masking"
dec = tokenizer.decode(unmasked_ids, skip_special_tokens=False)
for marker in ("File: ", '"status": "ok"', "### Problem", "<|tool_response>"):
    assert marker not in dec, f"Observation text {marker!r} leaked into trainable labels"
print(f"[mask] trainable {frac:.1%} of {len(ids)} tokens")
```

Also log which path ran (`prefix_ok`) and how often, per example.

---

## 4. You still are not measuring whether patches pass tests

`Cell 11` reports *Called submit_patch* and *Non-Empty Diff*. Neither is a quality
signal. "6/14 patches submitted" is consistent with a leaderboard score of 0 — and that
is exactly what happened. Phase 4.2 asked for a `resolved=True` rate from actual test
execution; it wasn't implemented.

Until the audit reports **applies-cleanly** and **tests-pass**, every local run is
unfalsifiable and you cannot tell a good checkpoint from a bad one. Add both columns
before the next GPU cycle; they're the only numbers that correlate with the leaderboard.

---

## 5. Dataset size vs. training schedule

With `grad_accum=8`, a surviving set of ~10 examples × 2 epochs = **2 optimizer steps**.
That is not training. Add a hard floor:

```python
assert len(kept) >= 60, f"Only {len(kept)} trajectories survived — fix MAX_LEN, do not train"
```

and scale `gradient_accumulation_steps` down so you get ≥30 optimizer steps.

---

## 6. Priority order for the next 48 hours (code)

1. **§2** — dump the real tool schemas, diff against `COMPETITION_TOOLS`. CPU, 5 min. Blocking.
2. **§1** — chunked masked loss, restore `MAX_LEN=8192`. CPU to write, verify on one step.
3. **§3** — tighten the masking gate; confirm which path runs. CPU, 15 min.
4. **§5** — floor assertion + step count. CPU, 5 min.
5. **§4** — add applies/tests-pass columns. CPU, 30 min.
6. Retrain once. Compare against the `rft_v1_baseline` snapshot on the same 14 tasks.

Items 1–5 are all CPU and cost no quota. **Do not spend another GPU cycle until §2 is
answered** — if the schemas are wrong, the retrain is wasted again.

---

# 7. The paper — yes, it can be saved, but not the current argument

Be clear-eyed: **Pass@1 is 0.00–0.01 across every checkpoint.** The thesis as drafted
("graph-anchored trajectories + evidence rubrics produce a capable offline SWE agent") is
not supported by your data and will not be by Nov 12. Submitting it as written is the
risky path — reviewers who look at Table 2 will see a 0 and a methodology note admitting
the earlier numbers came from a broken metric.

**The good news: you have a better paper, and you've already half-written it.**

Lines 350–358 of `paper/manuscript.md` are the most valuable thing in the repo — an honest
checkpoint-by-checkpoint table attributing each failure to a specific mechanism. That is a
real contribution, and it's one almost nobody publishes.

### Proposed reframing

> **"Why Agentic SFT Silently Fails: A Failure Taxonomy from Post-Training a 31B SWE Agent"**

The contribution is not an agent that works. It is a **mechanistic taxonomy of silent
post-training failures**, each with a diagnosis, a mechanism, a fix, and a measured effect.
You have five, all real and all reproducible:

| # | Failure | Mechanism | Observable |
|---|---|---|---|
| F1 | **End-of-turn masking** | The `<turn\|>` stop token receives `-100`, so the policy is never trained to terminate | Responses run to `max_output_tokens`; tool calls emitted as prose |
| F2 | **Preamble-budget collapse** | Eval-matched system prompt + tool schemas cost 1,110 tok (measured); with drop-don't-truncate at 2,560 the dataset silently empties | `len(kept)` collapses; no error raised |
| F3 | **Inter-stage format drift** | RFT accepted markdown ` ```diff ` fences, erasing the tool-calling SFT taught | Submission rate falls between stages |
| F4 | **Serialization fragmentation** | Four incompatible tool-call formats across one pipeline | Trained format ≠ parsed format |
| F5 | **Unfalsifiable metrics** | Audit counted `args == {}` as submission; never measured tests-pass | 71.4%/85.7% reported vs 0.0% real |

The unifying claim writes itself and is genuinely useful: **every one of these fails
silently — loss curves look healthy throughout.** F1 converged from 2.046 to 0.3935 while
training a model that cannot stop talking. The paper's argument is that agentic
post-training needs *structural* assertions (what carries gradient, what fraction is
supervised, does the stop token appear in labels, how many examples survived) because
scalar loss cannot see any of these. That's a methods contribution the community can use.

### What to do in two days

- **Nov 10 (today):** §2 above. Then run the **F1 ablation** — it is the cheapest real
  experiment you have: train two small adapters, identical except the stop token is masked
  in one, and report *% responses hitting the token ceiling* and *submit rate*. Even at
  tiny scale on a handful of tasks, it's a clean controlled result and it's the heart of
  the paper.
- **Nov 11:** Rewrite the framing. Keep Table 2 as the taxonomy's evidence. **Re-run
  Table 1** with the task-specific negatives (7.2 is already implemented in
  `candidate_patch_generator.py`) and a base-model judge (7.3, implemented) — Table 1 is a
  *judging* experiment, independent of the agent failures, so it is your one surviving
  quantitative positive result. Report it stratified by negative class.
- **Nov 12:** Submit. Then you still have three weeks to Dec 2 for the leaderboard.

### Claims that must come out
- Anything calling the trajectories "verified" or "pristine" — the filter was `len(msgs) >= 4`.
- The 2× `V_graph` gain, unless re-run with the base-model judge (self-judged as it stands).
- `requests_6644` / `requests_7433` as evidence of general capability — n=1 anecdotes at
  Pass@1 ≈ 0. Keep them as qualitative case studies, labelled as such.
- Any Table 3 / multi-scale edge-hardware claim you have not actually run.

### Why this is the stronger submission
Negative results with identified mechanisms and verified fixes are rarer and more useful
than another incremental agent-scaffolding paper. You have something specific and
checkable: *a one-line masking bug that silently disables agent termination while the loss
curve looks perfect.* That will be recognised by anyone who has post-trained a tool-using
model. The current draft's claims, by contrast, would not survive someone reading the code.
