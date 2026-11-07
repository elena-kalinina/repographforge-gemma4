# RepoGraphForge: Clean Master Pipeline Guide (Kaggle L4x4)

This guide organizes the entire RepoGraphForge post-training, evaluation, and submission pipeline into a clean, linear, error-proof sequence of notebook cells.

---

## 🧭 Master Cell Execution Map

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PART 1: SETUP & ENVIRONMENT                     │
│  [Cell 1] Section 1 from Ryan's Notebook (Install Dependencies)        │
│  [Cell 2] RepoGraphForge Patch: Fix Transformers & Disarm TorchAO       │
│  [Cell 3] Section 2 from Ryan's Notebook (Copy Dataset & Templates)    │
│  [Cell 4] Section 3 from Ryan's Notebook (Code Graph Utilities)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  PART 2: SFT POST-TRAINING (STAGE 4A)                  │
│  [Cell 5] RepoGraphForge: GPU VRAM Flush & Health Check                │
│  [Cell 6] RepoGraphForge: Write train_sft.py (Response-Masked SFT)     │
│  [Cell 7] RepoGraphForge: Launch SFT Training (~35 mins)               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│              PART 3: BEST-OF-N RFT POST-TRAINING (STAGE 4B)            │
│  [Cell 8] RepoGraphForge: Sync Trained SFT Adapter to sample_submission│
│  [Cell 9] Section 4 from Ryan's Notebook (Start vLLM Server)           │
│  [Cell 10] RepoGraphForge: Sample N=4 Rollouts & Filter with V_graph   │
│  [Cell 11] RepoGraphForge: Kill vLLM & Launch train_rft.py (~30s)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    PART 4: BENCHMARK & EVALUATION                      │
│  [Cell 12] RepoGraphForge: Sync Final RFT Adapter to sample_submission │
│  [Cell 13] Section 4 from Ryan's Notebook (Restart vLLM Server)        │
│  [Cell 14] Section 5 from Ryan's Notebook (Run 14-Task Holdout Sweep)  │
│  [Cell 15] RepoGraphForge: 2-Second Holdout Benchmark Audit Table      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                PART 5: PACKAGING & LEADERBOARD SUBMISSION              │
│  [Cell 16] Section 6 from Ryan's Notebook (Package submission.zip)     │
│  [Cell 17] RepoGraphForge: Direct Download Link / Submit to Leaderboard│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Cell-by-Cell Implementation

### PART 1: SETUP & ENVIRONMENT

#### Cell 1: `[Section 1 from Ryan's Notebook]`
*Run unchanged from Ryan Holbrook's starter notebook.*  
Installs offline dependencies (`swegemma`, `adk_submission`, `adk_eval_core`) from the attached wheelhouse dataset.

---

#### Cell 2: `[RepoGraphForge Patch]` Fix Transformers 5.13.1 & Disarm TorchAO
```python
import os, sys
from pathlib import Path

print("=== APPLYING REPO-GRAPH-FORGE ENVIRONMENT SAFEGUARDS ===")

# 1. Install transformers 5.13.1 and safetensors from wheelhouse
for p in [Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse"),
          Path("/kaggle/input/gemma-4-developer-agent-wheelhouse"),
          Path("/kaggle/input/competitions/gemma-4-developer-agent/wheels")]:
    if p.exists() and list(p.glob("*.whl")):
        !pip install -q --no-index --find-links={p} transformers safetensors
        print(f"✓ Installed transformers & safetensors from {p.name}")
        break

# 2. Patch upstream decorators bug in configuration_encoder_decoder.py
enc_file = Path("/usr/local/lib/python3.12/dist-packages/transformers/models/encoder_decoder/configuration_encoder_decoder.py")
if enc_file.exists():
    !sed -i 's/@auto_docstring.*//g' {enc_file}
    !sed -i 's/@strict.*//g' {enc_file}
    print("✓ Patched transformers encoder_decoder dataclass decorators.")

# 3. Disarm legacy torchao check in PEFT
!pip uninstall -y torchao 2>/dev/null
peft_import = Path("/usr/local/lib/python3.12/dist-packages/peft/import_utils.py")
if peft_import.exists():
    !sed -i 's/raise ImportError.*incompatible version of torchao.*/return False/g' {peft_import}
    print("✓ Patched PEFT torchao check.")
```

---

#### Cell 3: `[Section 2 from Ryan's Notebook]`
*Run unchanged from Ryan Holbrook's starter notebook.*  
Sets up `DATA_DIR`, `WORKING_DIR`, `AGENT_DIR`, and copies the base `sample_submission` template.

---

#### Cell 4: `[Section 3 from Ryan's Notebook]`
*Run unchanged from Ryan Holbrook's starter notebook.*  
Defines `GRAPH_DIR`, `EMBEDDINGS_DIR`, and code graph exploration utilities (`get_code_subgraph`, `get_code_neighbors`).

---

### PART 2: SFT POST-TRAINING (STAGE 4A)

#### Cell 5: `[RepoGraphForge]` GPU VRAM Flush & Health Check
```python
import os, torch

print("=== FLUSHING GPU MEMORY ===")
!fuser -k 8000/tcp 2>/dev/null
!pkill -9 -f vllm 2>/dev/null
!pkill -9 -f "Worker_TP" 2>/dev/null
!pkill -9 -f multiprocessing 2>/dev/null
torch.cuda.empty_cache()

num_gpus = torch.cuda.device_count()
print(f"✓ Detected {num_gpus} GPUs:")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

!nvidia-smi
```

---

#### Cell 6: `[RepoGraphForge SFT]` Write `train_sft.py` with Pure Native Tokenization
*(Note: `%%writefile` must be on Line 1 of this cell)*

```python
%%writefile /kaggle/working/train_sft.py
import os, json, logging
from pathlib import Path
import torch
import torch.nn as nn

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_sft")

import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False

from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2")
TRAJECTORIES_FILE = Path("/kaggle/working/data/trajectories/sft_combined.jsonl")
if not TRAJECTORIES_FILE.exists():
    TRAJECTORIES_FILE = Path("/kaggle/working/data/trajectories/sft_verified.jsonl")

OUTPUT_ADAPTER_DIR = Path("/kaggle/working/submission/adapters/main_lora")
OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Pure native tokenization with tool_calls mapped to real dicts
def format_chat_turns_native(example):
    raw_messages = example.get("messages", [])
    if not raw_messages:
        return {"text": example.get("text", "")}

    standardized = []
    last_tool_name = "get_code_subgraph"
    last_call_id = "call_0"

    for m in raw_messages:
        role = m.get("role", "user")
        content = m.get("content") or ""

        if role in ("assistant", "model"):
            role = "assistant"
            msg = {"role": role, "content": content}
            tcs = m.get("tool_calls", [])
            if tcs and isinstance(tcs, list):
                norm_tcs = []
                for idx, tc in enumerate(tcs):
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name") or tc.get("name", "tool")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: args = {"raw": args}
                    cid = tc.get("id") or f"call_{idx}"
                    last_tool_name = name
                    last_call_id = cid
                    norm_tcs.append({
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": args}
                    })
                msg["tool_calls"] = norm_tcs
            standardized.append(msg)

        elif role == "tool":
            standardized.append({
                "role": "tool",
                "name": m.get("name") or last_tool_name,
                "tool_call_id": m.get("tool_call_id") or last_call_id,
                "content": content
            })
        else:
            standardized.append({"role": "user", "content": content})

    text = tokenizer.apply_chat_template(standardized, tokenize=False)
    return {"text": text}

raw_ds = load_dataset("json", data_files=str(TRAJECTORIES_FILE), split="train")
formatted_ds = raw_ds.map(format_chat_turns_native)

# Assistant-only loss masking: mask user tokens with -100 so model never predicts <|turn>user
MAX_LEN = 1536
def tokenize_with_assistant_masking(example):
    text = example["text"]
    tokens = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
    input_ids = tokens["input_ids"]
    labels = list(input_ids)

    # In Gemma 4 chat template: <|turn>model indicates assistant response
    model_turn_id = tokenizer.convert_tokens_to_ids("<|turn>")
    turn_end_id = tokenizer.convert_tokens_to_ids("<turn|>")

    in_assistant_turn = False
    for i in range(len(input_ids)):
        if input_ids[i] == model_turn_id:
            in_assistant_turn = True
        elif input_ids[i] == turn_end_id and in_assistant_turn:
            in_assistant_turn = False
        
        if not in_assistant_turn:
            labels[i] = -100  # Mask loss on user/system tokens!

    tokens["labels"] = labels
    return tokens

tokenized_ds = formatted_ds.map(tokenize_with_assistant_masking, remove_columns=formatted_ds.column_names)
logger.info("✓ Prepared %d trajectories with assistant-only loss masking.", len(tokenized_ds))

# Balanced GPU allocation across 4x L4 GPUs
num_gpus = torch.cuda.device_count()
max_memory = {i: "16.5GiB" for i in range(num_gpus)}

base_model = Gemma4ForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory=max_memory,
    local_files_only=True,
)

TEXT_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
text_linear_targets = [
    name for name, mod in base_model.named_modules()
    if name.endswith(TEXT_SUFFIXES)
    and isinstance(mod, nn.Linear)
    and "vision" not in name.lower()
    and "audio" not in name.lower()
]

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=text_linear_targets,
)

base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()
model = get_peft_model(base_model, peft_config)
logger.info("✓ LoRA Adapter Attached across %d text linear targets:", len(text_linear_targets))
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="/kaggle/working/sft_checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="no",
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_ds,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
)

logger.info("🚀 STARTING SFT TRAINING (3 Epochs, %d steps)...", len(tokenized_ds) * 3 // 8)
trainer.train()

trainer.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))

for f in OUTPUT_ADAPTER_DIR.glob("*.jinja"): f.unlink()
for f in OUTPUT_ADAPTER_DIR.glob("README.md"): f.unlink()

logger.info("✓ SFT TRAINING COMPLETE! Adapter exported to %s", OUTPUT_ADAPTER_DIR)
```

---

#### Cell 7: `[RepoGraphForge SFT Execution]` Launch Training
```python
!python3 /kaggle/working/train_sft.py
```

---

### PART 3: BEST-OF-N RFT POST-TRAINING (STAGE 4B)

#### Cell 8: `[RepoGraphForge Sync]` Sync Adapter & Set Anti-Hesitation Gate
```python
import shutil
from pathlib import Path
from adk_submission import discover_adapters, discover_declared_models

WORKING_DIR = Path("/kaggle/working")
AGENT_DIR = WORKING_DIR / "sample_submission"
src_adapter = WORKING_DIR / "submission/adapters/main_lora"
dst_adapter = AGENT_DIR / "adapters/main_lora"

# 1. Sync adapter into sample_submission
dst_adapter.mkdir(parents=True, exist_ok=True)
for f in src_adapter.glob("*"):
    if not f.name.endswith(".jinja") and not f.name.endswith(".md"):
        shutil.copy2(f, dst_adapter / f.name)

# 2. Sanitize
for p in [src_adapter, dst_adapter]:
    for f in p.glob("*.jinja"): f.unlink()
    for f in p.glob("README.md"): f.unlink()

# 3. Inject Anti-Hesitation Prompt
prompt_file = AGENT_DIR / "prompts/system.md"
prompt_file.parent.mkdir(parents=True, exist_ok=True)
prompt_file.write_text("""You are an autonomous software engineering agent equipped with a pre-computed repository code graph.

MANDATORY EXECUTION CONSTRAINTS:
1. MAX 1 FILE READ: Once you read the primary file where the bug occurs (e.g. models.py), you are STRICTLY FORBIDDEN from reading secondary helper files, compatibility shims (like compat.py), or test files.
2. DO NOT SECOND-GUESS: Do not over-analyze unrelated symbols or utility functions. Focus ONLY on the exact symptom reported in the problem statement.
3. MANDATORY EDIT ON TURN 3: You MUST invoke `edit_file` immediately after reading the target function. Never emit thoughts without a tool call.
4. IMMEDIATE SUBMISSION: As soon as `edit_file` succeeds, your next turn MUST be `submit_patch`.

EXECUTION SEQUENCE:
Turn 1: `get_code_subgraph` to locate the function.
Turn 2: `read_file` on that function's file.
Turn 3: `edit_file` to apply the fix.
Turn 4: `submit_patch` to finish.
""".strip())

# 4. Cap output tokens to 1024
sampling_file = AGENT_DIR / "configs/sampling.yaml"
sampling_file.parent.mkdir(parents=True, exist_ok=True)
sampling_file.write_text("temperature: 0.1\ntop_p: 0.95\nmax_output_tokens: 1024\n")

# 5. Configure agent.yaml
agent_yaml = AGENT_DIR / "agent.yaml"
agent_yaml.write_text("""name: swe_baseline_agent
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include configs/sampling.yaml
""".strip())

print("=== SYNC COMPLETE ===")
print(f"✓ Adapter Size:     {(dst_adapter / 'adapter_model.safetensors').stat().st_size / 1e6:.2f} MB")
print(f"✓ Declared Model:   {discover_declared_models(AGENT_DIR)}")
print(f"✓ Declared Adapter: {discover_adapters(AGENT_DIR)}")
print("=" * 45)
```

---

#### Cell 9: `[Section 4 from Ryan's Notebook]`
*Run unchanged from Ryan Holbrook's starter notebook.*  
Boots the vLLM server serving `main_lora` across 4x L4 GPUs on port 8000. Wait ~2 minutes until it logs `vLLM server started...`.

---

#### Cell 10: `[RepoGraphForge]` Sample N=4 Rollouts & Filter with $V_{\text{graph}}$ (~2.5 mins)
```python
import json, re, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from transformers import AutoTokenizer

print("=== SAMPLING FRESH RFT ROLLOUTS (N=4) ===")
BASE_URL = "http://127.0.0.1:8000/v1"
SYNTHETIC_TASKS_FILE = Path("/kaggle/working/data/synthetic_tasks/synthetic_tasks.jsonl")
MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2")
RFT_SPLITS_DIR = Path("/kaggle/working/data/trajectories/rft_splits")
RFT_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

tasks = []
with open(SYNTHETIC_TASKS_FILE) as f:
    for line in f:
        if line.strip(): tasks.append(json.loads(line))
tasks = tasks[:50]

def call_vllm_messages(messages, temp=0.7):
    payload = {"model": "main_lora", "messages": messages, "temperature": temp, "max_tokens": 1024}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
    except Exception: return None

def extract_patch(text):
    m = re.search(r"```(?:diff|patch)?\n([\s\S]*?)```", text)
    if m: return m.group(1).strip()
    m2 = re.search(r"(diff --git a/[\s\S]+)", text)
    if m2: return m2.group(1).strip()
    return ""

def score_v_graph(prob, patch, rubric):
    if not patch or len(patch) < 30: return False
    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric))
    sys_p = "You are an evidence-anchored code verification judge (RepoGraphForge V_graph). Evaluate the candidate patch strictly against the provided structural evidence rubric. If ANY contract is violated, omitted, or partially implemented, output FAIL. Respond strictly with JSON: {\"verdict\": \"PASS\" | \"FAIL\"}"
    user_p = f"### Problem Statement:\n{prob[:1800]}\n\n### Evidence Rubric:\n{rubric_text}\n\n### Candidate Patch:\n{patch[:2000]}\n"
    res = call_vllm_messages([{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}], temp=0.0)
    return bool(res and '"PASS"' in res.upper())

def sample_task(task):
    syn_id = task.get("synthetic_id", "syn")
    prob = task.get("problem_statement", "")
    anchor = task.get("anchor_node", "core")
    rubric = task.get("evidence_rubric", [])
    prompt = f"### Task ID: {syn_id}\n### Anchor Node: {anchor}\n\n### Problem Statement:\n{prob}\n"

    samples = []
    for _ in range(4):
        msgs = [{"role": "user", "content": prompt}]
        t1 = call_vllm_messages(msgs, temp=0.7)
        if not t1: continue
        msgs.append({"role": "assistant", "content": t1})

        obs = f"<|tool_response>response:get_code_subgraph(anchor='{anchor}'):\nStatus: OK\nNow apply the fix via edit_file and call submit_patch.<tool_response|>"
        msgs.append({"role": "user", "content": obs})

        t2 = call_vllm_messages(msgs, temp=0.7)
        if not t2: continue

        patch = extract_patch(t2)
        v_graph = score_v_graph(prob, patch, rubric)

        standardized = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": t1},
            {"role": "user", "content": obs},
            {"role": "assistant", "content": t2}
        ]
        try:
            rendered = tokenizer.apply_chat_template(standardized, tokenize=False)
        except Exception:
            rendered = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n{t1}<turn|>\n<|turn>user\n{obs}<turn|>\n<|turn>model\n{t2}<turn|>"

        samples.append({"synthetic_id": syn_id, "text": rendered, "v_graph_pass": v_graph})
    return samples

t0 = time.time()
all_candidates = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(sample_task, t): t for t in tasks}
    for i, fut in enumerate(as_completed(futures), start=1):
        all_candidates.extend(fut.result())
        if i % 10 == 0 or i == len(tasks):
            print(f"  [{i}/{len(tasks)}] sampled ({len(all_candidates)} rollouts in {time.time()-t0:.1f}s)...")

by_task = defaultdict(list)
for c in all_candidates: by_task[c["synthetic_id"]].append(c)

passing = []
for syn_id, rollouts in by_task.items():
    passed = [r for r in rollouts if r["v_graph_pass"]]
    if passed:
        passing.append(min(passed, key=lambda r: len(r["text"])))

out_file = RFT_SPLITS_DIR / "rft_v_graph_accepted.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    for p in passing:
        f.write(json.dumps({"synthetic_id": p["synthetic_id"], "text": p["text"]}) + "\n")

print(f"✓ Saved {len(passing)} / {len(tasks)} passing trajectories to {out_file.name}")
```

---

#### Cell 11: `[RepoGraphForge]` Kill vLLM & Launch RFT Alignment Pass (~30s)
```python
# 1. Free VRAM
!fuser -k 8000/tcp 2>/dev/null
!pkill -9 -f vllm 2>/dev/null
!pkill -9 -f Worker_TP 2>/dev/null
import torch
torch.cuda.empty_cache()

# 2. Write and execute train_rft.py
%%writefile /kaggle/working/train_rft.py
import os, json, logging
from pathlib import Path
import torch
import torch.nn as nn

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_rft")

import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False

from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset

MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2")
SFT_ADAPTER_DIR = Path("/kaggle/working/submission/adapters/main_lora")
RFT_DATASET_FILE = Path("/kaggle/working/data/trajectories/rft_splits/rft_v_graph_accepted.jsonl")
OUTPUT_ADAPTER_DIR = Path("/kaggle/working/submission/adapters/main_lora")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

raw_ds = load_dataset("json", data_files=str(RFT_DATASET_FILE), split="train")

def tokenize_fn(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=1536, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_ds = raw_ds.map(tokenize_fn, remove_columns=raw_ds.column_names)
logger.info("Loaded %d verifier-curated Best-of-N trajectories.", len(tokenized_ds))

num_gpus = torch.cuda.device_count()
max_memory = {i: "16.5GiB" for i in range(num_gpus)}

base_model = Gemma4ForConditionalGeneration.from_pretrained(
    str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map="balanced", max_memory=max_memory, local_files_only=True
)

model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER_DIR), is_trainable=True)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

effective_grad_accum = min(2, max(1, len(tokenized_ds) // 2))
training_args = TrainingArguments(
    output_dir="/kaggle/working/rft_checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=effective_grad_accum,
    learning_rate=5e-5,
    num_train_epochs=2,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(model=model, train_dataset=tokenized_ds, args=training_args, data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8))
logger.info("🚀 STARTING RFT ALIGNMENT PASS (2 Epochs, lr=5e-5)...")
trainer.train()

trainer.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))
for f in OUTPUT_ADAPTER_DIR.glob("*.jinja"): f.unlink()
for f in OUTPUT_ADAPTER_DIR.glob("README.md"): f.unlink()
logger.info("✓ RFT ALIGNMENT COMPLETE! Final policy adapter is ready.")

# 3. Launch RFT script
!python3 /kaggle/working/train_rft.py
```

---

### PART 4: BENCHMARK & EVALUATION

#### Cell 12: `[RepoGraphForge]` Re-Run Sync Cell
*Re-run Cell 8 to sync the freshly aligned RFT adapter into `sample_submission`.*

---

#### Cell 13: `[Section 4 from Ryan's Notebook]`
*Click Run on Section 4 to start vLLM serving the aligned adapter on port 8000.*

---

#### Cell 14: `[Section 5 from Ryan's Notebook]` Configure & Run Evaluation Loop
```python
# In Section 5:
TASKS_PATH = Path('/kaggle/working/data/splits/val_ood_tasks.jsonl')
tasks = load_tasks(TASKS_PATH)
SAMPLE_TASKS = tasks

timeout_seconds = 300
verification_timeout_seconds = 300
max_turns = 12
```
*Click Run on the evaluation loop cell directly below it!*

---

#### Cell 15: `[RepoGraphForge]` 2-Second Holdout Benchmark Audit Table
```python
import json
from pathlib import Path
import pandas as pd

TRACES_DIR = Path("/kaggle/working/results/traces")
records = []

for p in sorted(TRACES_DIR.glob("trace_*.json")):
    data = json.loads(p.read_text())
    t_id = p.stem.replace("trace_", "")
    steps = data.get("steps", [])
    metrics = data.get("final_metrics", {})

    tool_call_count = 0
    files_edited = set()
    submitted = False
    
    for s in steps:
        for tc in s.get("tool_calls", []):
            tool_call_count += 1
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: pass
            if isinstance(args, dict):
                if "old_string" in args and "filepath" in args:
                    files_edited.add(args["filepath"])
                elif args == {} or fn.get("name") == "submit_patch":
                    submitted = True

    records.append({
        "Task ID": t_id,
        "Tool Calls": tool_call_count,
        "Patch Submitted": "✓ YES" if submitted else "❌ NO_PATCH",
        "Files Edited": list(files_edited) if files_edited else "None",
        "Total Tokens": metrics.get("total_tokens", 0),
    })

df = pd.DataFrame(records)
display(df)
print(f"Patch Submission Rate: {sum(df['Patch Submitted'] == '✓ YES')} / {len(df)} ({sum(df['Patch Submitted'] == '✓ YES')/len(df)*100:.1f}%)")
```

---

### PART 5: PACKAGING & LEADERBOARD SUBMISSION

#### Cell 16: `[Section 6 from Ryan's Notebook]` Package `submission.zip`
*Run unchanged from Ryan's notebook to create and validate `/kaggle/working/submission.zip`.*

---

#### Cell 17: `[RepoGraphForge]` Direct Download Link
```python
from IPython.display import FileLink
FileLink("submission.zip")
```
*Click the blue link to download `submission.zip` directly to your laptop, then upload it to the competition page to claim your daily submission slot!*
