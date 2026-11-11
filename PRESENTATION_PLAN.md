# Presentation Plan — Notebook Demo for Coding-Agent Practitioners

Date: shortly before Dec 2. Format: a live notebook, not slides. Audience: people who
build or evaluate coding agents. Deliverable now: the abstract (§1). Everything else is
the build plan.

Design rule: **three acts, each independently runnable, each with a pre-recorded playback
mode.** A live model is a toggle, never a dependency. If the Dec 2 experiment slips, Act 3
collapses to a two-minute epilogue and the talk still works.

---

## 1. Abstract (submit this)

> **Code graphs as teacher, tool, and judge: what post-training Gemma 4 for repository
> bug-fixing actually taught us**
>
> We set out to turn pre-computed repository code graphs into a post-training engine for
> an offline Gemma 4 coding agent: sampling multi-hop subgraphs to synthesize cross-file
> bug tasks, compiling structural "evidence rubrics" to judge patches without running
> tests, and using those rubrics to filter rejection fine-tuning. Over seven training runs
> the loss converged every time and the agent resolved almost nothing. This talk is a
> live, notebook-driven account of both halves. First, the graph as a *tool*: on a real
> `psf/requests` issue whose symptom and fix live in different files, we run a grep-first
> and a graph-first agent side by side and watch what graph localization costs and saves
> in tokens, turns, and time. Second, the graph as a *judge*: we show an unanchored LLM
> judge and a rubric-anchored judge disagreeing on gold, truncated, and plausible-but-wrong
> patches, then let `pytest` settle it; where our Dec 2 experiment permits, we show the
> same selection on real agent rollouts with real test outcomes. Third, the part nobody
> publishes: six ways the training pipeline failed while reporting success, each invisible
> to the loss curve because it lives in a handful of supervised tokens, including tool
> declarations hand-copied from the harness that had quietly drifted from it. We leave the
> audience with a short set of structural assertions to run before any agentic
> fine-tuning job, and a candid answer to the question we started with: whether code-graph
> anchoring helps an agent at all.

Shorter variant (if a 100-word limit applies):

> We tried to post-train Gemma 4 into an offline repository bug-fixing agent using code
> graphs three ways: to synthesize tasks, to navigate, and to judge patches without
> running tests. The loss converged in every run; the agent resolved almost nothing. In a
> live notebook we show a graph-first agent against a grep-first one on a real cross-file
> `requests` bug, two LLM judges disagreeing until `pytest` settles it, and the six silent
> pipeline failures a loss curve cannot see. You leave with a checklist of structural
> assertions to run before your next agentic fine-tuning job.

---

## 2. The three acts

### Act 1 — The graph as a tool (8 min)

One task, two agents, same model, same tool schemas, same budget.

- Task: `requests_7502` (symptom in `models.py`, fix in `utils.py`) — the cross-file case.
  Backup: `requests_7505` (single-file, resolved once in 53 s).
- Agent A: tools = `read_file`, `edit_file`, `run_command`, `submit_patch` only (grep-first).
- Agent B: same plus `get_code_subgraph`, `get_code_neighbors`, `search_similar_code`,
  with the graph-first budget prompt.
- Show per turn: tool called, tokens in / out, cumulative tokens, wall-clock; final diff;
  `pytest` result on the task's test.
- Punchline is whatever the numbers say. Expected: B localizes in 1–2 calls, A pages the
  caller file; B may still fail the fix. Say so.

### Act 2 — The graph as a judge (8 min)

- One task, three candidate patches: gold maintainer fix; truncated fix (first half of the
  hunk); drift patch (touches the right file, adds a no-op guard).
- Ask $V_{\text{free}}$ (issue + diff) and $V_{\text{graph}}$ (issue + diff + rubric).
  Show reasoning text, not just verdicts.
- Then `git apply` + `pytest` on each. The truncated patch will usually not even apply —
  point out that the pilot study counted it as a "negative" without checking.
- Show the rubric and where it came from (gold patch → leakage). Then the no-gold rubric
  if `EXPERIMENT_PLAN_DEC2.md` Stage 4.2 ran.

### Act 3 — Real rollouts, real tests (8 min, or 2 min epilogue)

- **If the Dec 2 experiment landed:** for one mixed-outcome task, show $N=4$ real rollouts
  from the sandbox, their actual test verdicts, and which rollout each judge would have
  picked. Then the selection table (pass@1 / V_free / V_graph / oracle pass@4).
- **If it did not:** render one training trajectory through the real Gemma 4 chat
  template, colour the supervised tokens, switch the old masking loop on, and watch the
  stop token go dark while the computed loss changes by ~1%. Then show Gate 8 catching
  `node_id` against the harness signature.

### Close (3 min)

The nine gates as a single scrolling cell of assertions. The honest answer to "does the
graph help": whatever Table B in the Dec 2 addendum says.

---

## 3. Build plan

### 3.1 Repository layout

```
demo/
├── demo.ipynb                 # the talk; every act has a PLAYBACK flag
├── local_harness.py           # the nine tools reimplemented locally (~200 lines)
├── judges.py                  # V_free / V_graph prompts, thin client over Ollama or vLLM
├── record.py                  # runs an agent live and writes a trace JSON
├── data/
│   ├── tasks_demo.jsonl       # 2–3 tasks pulled from the competition tasks.jsonl
│   ├── graphs/                # the matching <repo>_<commit>.json graph files
│   └── rubrics_demo.jsonl     # gold-derived and no-gold rubrics for the demo tasks
├── repos/                     # psf/requests checkouts at base_commit (gitignored)
└── traces/                    # recorded traces for playback (committed)
```

### 3.2 Steps

**Nov 13–14 — data and skeleton (local, 3 h)**
- [ ] From the Kaggle dataset download only: `tasks.jsonl`, the graph JSONs for the 2–3
      demo tasks' `base_commit`s (`requests_<commit>.json`), and the matching `.npz` if
      present. Total is small; do not pull the 22 GB.
- [ ] `git clone https://github.com/psf/requests demo/repos/requests && git checkout <base_commit>`
      for each demo task. Create a venv per checkout with `pip install -e .` and `pytest`.
      Apply each task's `test_patch` into a scratch copy and run the named test to confirm
      it **fails** on base and **passes** with the gold patch. If not, pick another task.
- [ ] Notebook skeleton: title cell, three act headers, a `PLAYBACK = True` constant,
      a trace-rendering helper (turn table + token bars + final diff with syntax colours).

**Nov 15–17 — local harness and model (local, 1–2 days)**
- [ ] `local_harness.py`: implement the nine tools with the **exact signatures** in
      `src/chat_format.py` `_FALLBACK_COMPETITION_TOOLS`:
      - `read_file(filepath, start_line=None, end_line=None)`, `edit_file(filepath,
        old_string, new_string, allow_multiple=False)`, `write_file`, `run_command`
        (subprocess in the checkout, 60 s timeout), `get_status` (git status --short),
        `submit_patch()` → `git diff`.
      - `get_code_subgraph(nodes)`, `get_code_neighbors(node, edge_type=None,
        max_neighbors=50)` over the networkx graph via `src/data_loader.RepoGraph`;
        suffix matching on dotted ids as the harness documents.
      - `search_similar_code(query, k=10)`: the competition embeds queries with an
        unknown encoder, so use a lexical fallback (token overlap on node id + text).
        Label it as such in the notebook.
      - Return observation strings in the **real harness format** captured in
        `experiments/logs/real_observations.json` (from `EXPERIMENT_PLAN_DEC2.md`
        Stage 1.2). If that capture does not exist yet, use a plain format and note it.
- [ ] Model serving on the Mac: install Ollama; check which Gemma 4 sizes are published
      (`ollama search gemma4` / the Ollama library page) and pull the largest that fits
      RAM with headroom (E4B first; 12B if ≥24 GB). Confirm tool calling works with a
      one-tool smoke test through the OpenAI-compatible endpoint
      (`http://localhost:11434/v1/chat/completions` with `tools=[...]`).
- [ ] Agent loop in `record.py`: system prompt (`chat_format.DEFAULT_SYSTEM_PROMPT` or
      the grep-first variant), tool schemas from `chat_format.COMPETITION_TOOLS`
      (filtered for Agent A), budget of 15 calls, `temperature 0.1`, writes
      `traces/<task>_<agent>_<model>.json` with per-turn tokens and timings.
- [ ] Fallback if local serving is too slow or tool calling is flaky: run `record.py`
      against the Kaggle vLLM endpoint from inside a Kaggle notebook (the 31B, real
      harness) and bring the trace files home. Playback is identical either way.

**Nov 18–22 — record Acts 1 and 2 (mixed, 1 day)**
- [ ] Record Agent A and Agent B on `requests_7502` and `requests_7505`, 3 runs each,
      keep the most representative trace per pair (and say "representative of 3").
- [ ] Run each final diff through `git apply --check` + the task test; store the verdict
      in the trace.
- [ ] Act 2: build the three candidate patches with
      `src/candidate_patch_generator.py` functions; generate the gold rubric with
      `src/rubric_generator.py` and the no-gold rubric (`--no-gold`, Stage 0.6 of the
      experiment plan); run both judges with the demo model and with the 31B (Kaggle) and
      record both; run `pytest` on all three patches. Store everything in
      `traces/act2_<task>.json`.
- [ ] Wire playback: each act reads its trace and renders. Full dry run in playback mode
      end to end, < 25 min.

**Nov 25–29 — Act 3 and polish (local, 1 day)**
- [ ] If `EXPERIMENT_PLAN_DEC2.md` Stage 7.2 exported `demo/traces/act3_*.json`, build the
      rollout comparison view and the selection table cell.
- [ ] Otherwise build the epilogue: load the Gemma 4 tokenizer (needs HF access to the
      gated repo; alternatively copy the tokenizer files from Kaggle), render one
      trajectory with `chat_format.build_example`, colour supervised tokens with
      `IPython.display.HTML`, re-run with the old six-line loop from `DIAGNOSIS.md` B1,
      show the stop token flipping to masked, and compute the fraction of supervised
      tokens it represents.
- [ ] Gates cell: the nine assertions in one cell, run live against the demo trajectory
      and `COMPETITION_TOOLS`; make one of them fail on purpose (`node_id`) and show the
      refusal message.
- [ ] Two full dry runs: one with `PLAYBACK = True`, one live. Time both. Decide which
      acts go live on the day; default is playback with live Act 2 (cheapest and most
      reliable live moment).

**Dec 1 — freeze**
- [ ] Commit `demo/` with traces. Export the notebook to HTML as a backup if the kernel
      misbehaves on the day.

### 3.3 Risks and mitigations

| Risk | Mitigation |
| :--- | :--- |
| Local Gemma 4 tool calling unreliable through Ollama | Record on Kaggle vLLM; play back locally |
| `requests` tests slow or flaky at old commits | Pre-build venvs; run only the task's named test; cache results in the trace |
| Dec 2 experiment slips | Act 3 epilogue is fully independent and CPU-only |
| Gated tokenizer download fails on the day | Vendor tokenizer files into `demo/data/tokenizer/` (check licence allows redistribution; otherwise keep local only) |
| Live model gives a boring or wrong answer | That is the talk's honesty point; have the recorded run one toggle away |

### 3.4 What the demo must not do

- Claim a resolution rate from Act 1 (n = 1–2 tasks). Show the traces; let the Dec 2
  table carry the numbers.
- Hide that Agent B's rubric saw the gold patch unless the no-gold arm is shown alongside.
- Run anything on stage that has not been run in the dry run.
