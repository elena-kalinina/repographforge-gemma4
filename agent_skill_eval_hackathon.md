# Agent Skill Evaluation & Regression Prevention: 90-Minute Hackathon Blueprint

> **Context:** In a multi-team engineering organization, independent teams build and expose tools, APIs, and agent skills (MCP servers, Agent Development Kit configs, OpenAPI specs). When a team adds or modifies a skill, changes can break the autonomous agent in subtle, non-deterministic ways. This blueprint provides a hands-on, 90-minute hackathon challenge to build an automated regression testbed for agent skills.

---

## 1. The Core Problem: Why Agents Break on Skill Updates

Traditional software uses strict compile-time types and unit tests. In contrast, AI agents interact with tools through natural language descriptions and runtime JSON interfaces. When a skill is updated, failures usually fall into four categories:

| Failure Mode | Mechanism | Symptom in Production |
| :--- | :--- | :--- |
| **1. Semantic Contract Drift** | The API still returns `200 OK`, but field semantics or formats changed (e.g., date formats, renamed keys, altered pagination). | Agent hallucinates or passes bad data to downstream steps without crashing. |
| **2. Prompt / Routing Hijacking** | A skill update introduces overly broad wording (e.g., *"Use this tool for all customer inquiries"*). | **Prompt Bleed**: The agent stops calling other tools and inappropriately routes requests to the updated tool. |
| **3. Sycophantic Silent Failure** | A tool returns an error code or empty payload, but the agent sycophantically reports success to the user. | User is told *"Action completed!"* while no backend modification occurred. |
| **4. Incomplete / Vague Guidance** | The skill description lacks negative boundaries, expected argument units, or required preconditions. | Agent enters retry loops, exhibits high routing variance, or hallucinates parameters. |

---

## 2. The 90-Minute Hackathon Schedule

```
┌─────────────────┬─────────────────────────────────────────────────────────┐
│ 00:00 – 00:15   │ Kickoff & Live Failure Demonstration                    │
│ 00:15 – 01:05   │ Hands-on Build (Teams choose 1 Evaluation Tier)         │
│ 01:05 – 01:25   │ Red Team vs. Blue Team Gauntlet (Break & Defend)        │
│ 01:25 – 01:30   │ Synthesis & CI/CD Action Plan                           │
└─────────────────┴─────────────────────────────────────────────────────────┘
```

---

## 3. The 5 Evaluation Tracks (Hackathon Challenge Tracks)

Participants divide into teams. Each team picks one track to build during the 50-minute implementation sprint:

### Track 1: Deterministic Schema & Invariant Linter ($V_{\text{schema}}$)
* **Objective:** Ensure mechanical contract safety before the agent is even invoked.
* **Challenge Tasks:**
  1. **Breaking Schema Changes:** Detect removed fields, newly added non-optional parameters, or altered return payload types.
  2. **Three-Point Invariant Enforcer:** Enforce that every skill PR defines:
     - **Input Invariant:** (e.g., `account_id` regex validation).
     - **Output Invariant:** (e.g., standard `{ status: "success" | "error" }` wrapper).
     - **Error Invariant:** (e.g., 404 returns structured JSON, never HTML or raw stack traces).

### Track 2: Prompt Ambiguity & Routing Entropy Benchmark ($V_{\text{clarity}}$)
* **Objective:** Detect vague, under-specified, or ambiguous wording in skill prompts that causes erratic tool selection.
* **Challenge Tasks:**
  1. **Temperature & Seed Stability Test:** Run 10 canonical queries at $T = 0.7$ with 10 different random seeds. Calculate **Tool Selection Entropy**.
     - *Pass:* Consistent tool selection across all seeds ($\ge 95\%$).
     - *Fail:* Model flips choices (e.g., picks Tool A 6 times, Tool B 3 times, asks user 1 time) $\rightarrow$ flags prompt as ambiguous.
  2. **Paraphrase Sensitivity Test:** Feed 5 linguistic variations per intent (terse, verbose, slang, typo-tolerant). Assert trigger recall $\ge 90\%$.
  3. **Turn Inflation Monitor:** Run benchmark tasks and measure turns/tool calls. If average turns increase significantly ($\text{Turns}_{\text{new}} / \text{Turns}_{\text{baseline}} > 1.5$), flag the prompt for low information scent / missing argument guidance.

### Track 3: Routing Collision & Semantic Overlap Benchmark ($V_{\text{routing}}$)
* **Objective:** Prevent a modified tool description from stealing traffic from other teams' tools (Prompt Bleed).
* **Challenge Tasks:**
  1. **Embedding Cosine Distance Check:** Compare the new skill description against all existing skills. Flag collisions where $\cos(\mathbf{z}_{\text{new}}, \mathbf{z}_{\text{existing}}) > 0.85$.
  2. **Zero-Shot Routing Regression Test:** Run 25 golden user queries across unrelated domains (Billing, Auth, DevOps, CRM). Assert that 0% of unrelated queries flip to the newly modified tool.
  3. **Negative Boundary Probing:** Feed "near-miss" queries (queries using related vocabulary but out of scope). Assert 0% false-positive triggering.

### Track 4: Chaos & Fault Injection Testing ($V_{\text{chaos}}$)
* **Objective:** Test how the agent behaves when the tool fails, degrades, or is rate-limited.
* **Challenge Tasks:** Mock the tool backend and inject 4 deterministic perturbations:
  1. *The 429 Rate Limit:* Does the agent inform the user or sleep, rather than entering an infinite retry loop?
  2. *The Empty List `[]`:* Does the agent acknowledge absence of data, or hallucinate records?
  3. *The Missing Permission (403):* Does the agent explain permission boundaries cleanly?
  4. *The Malformed JSON:* Does the agent handle parser recovery gracefully?

### Track 5: Sycophancy & Evidence Rubric Verifier ($V_{\text{contract}}$)
* **Objective:** Prevent the agent from claiming success when the backend action failed.
* **Challenge Tasks (Borrowed from RepoGraphForge $V_{\text{graph}}$):**
  1. Do not ask an unanchored LLM: *"Did this look good?"* (LLMs sycophantically approve broken runs).
  2. Define an **Evidence Rubric**: Check the actual tool call parameters and response status.
  3. If tool status is `error` but the final assistant message contains affirmative language (*"Successfully updated"*, *"Done"*), trigger an **Immediate Failure (Sycophantic Drift)**.

---

## 4. Testing Vague, Incomplete, or Ambiguous Skill Prompts

When a skill breaks not because of a code bug, but because the **natural language wording in the prompt is vague or missing context**, deterministic unit tests will not catch it. Here are four concrete techniques to test prompt wording quality:

### A. Routing Stability Across Temperature & Seeds (Entropy Test)
* **Principle:** A crisp, well-specified tool description will be selected deterministically. A vague description causes the model to guess randomly.
* **Test:** Run the same user prompt 10 times at $T = 0.7$ with different random seeds.
* **Metric:** **Tool Selection Entropy**. If the model picks the tool 10/10 times, the prompt has high clarity. If it flips between tools (e.g., 6 times Tool A, 4 times Tool B), the prompt wording is ambiguous and must be refined with clearer trigger boundaries.

### B. Paraphrase Sensitivity / Intent Recall
* **Principle:** A tool description that is too narrow or relies on specific keywords will fail when users phrase requests differently.
* **Test:** Take 5 core user intents and generate 5 linguistic variations each (terse, verbose, slang, question format, typo-tolerant).
* **Metric:** **Trigger Recall across Paraphrases** ($\ge 90\%$). If the tool only triggers on exact keyword matches, the prompt lacks semantic generality.

### C. Trajectory Inefficiency & Turn Inflation
* **Principle:** When a skill prompt fails to clearly document required arguments, preconditions, or default behaviors, the agent does not immediately fail—it wanders.
* **Test:** Measure average turns and tool calls required to complete standard tasks.
* **Metric:** **Turn Inflation Ratio** $\left(\frac{\text{Turns}_{\text{new}}}{\text{Turns}_{\text{baseline}}}\right)$. If updating a skill prompt increases the average turns from 2.2 to 5.1 due to tool call retries and parameter corrections, the prompt information scent is degraded.

### D. The "Negative Boundary" Probe (Over-Triggering)
* **Principle:** Every good skill prompt must specify when **NOT** to trigger.
* **Test:** Feed the agent prompts that sound superficially related to the tool's domain, but are explicitly meant for other systems or human escalation.
* **Metric:** **False Positive Trigger Rate** on negative examples.

---

## 5. Summary Deliverable: The PR Gate (CI/CD Action)

At the end of the hackathon, teams combine their checks into a single pull-request validation workflow:

```yaml
# .github/workflows/skill_eval.yaml
name: Skill Regression & Contract Gate
on: [pull_request]

jobs:
  evaluate-skill:
    runs-on: ubuntu-latest
    steps:
      - name: Tier 1 - Schema & Invariant Linting
        run: python -m eval.schema_linter --skills-dir ./skills
      
      - name: Tier 2 - Prompt Ambiguity & Entropy Check
        run: python -m eval.check_prompt_clarity --trials 10 --temp 0.7
      
      - name: Tier 3 - Description Semantic Collision Check
        run: python -m eval.check_prompt_bleed --threshold 0.85
      
      - name: Tier 4 - Routing Stability Benchmark
        run: python -m eval.run_routing_bench --trials 10 --temp 0.7
      
      - name: Tier 5 - Sycophantic Drift & Chaos Verification
        run: python -m eval.run_chaos_gauntlet
```
