#!/usr/bin/env python3
"""
Milestone 4 (Phase 1, 3 & 4 of `FIX_PLAN.md`): Fine-tune Gemma 4 using LoRA on RepoGraphForge trajectories
and export the adapter weights directly to submission/adapters/main_lora/.

Uses the single canonical `src/chat_format.py` module (`FIX_PLAN.md` Phase 1.5 & 3.1):
  - B1: Exact turn-by-turn prefix-diff assistant masking INCLUSIVE of `<turn|>` / `<end_of_turn>`.
  - B2: `MAX_LEN = 8192` with strict drop-don't-truncate policy (logs dropped count).
  - B4: Single-BOS hygiene (`add_special_tokens=False`, asserts `bos_count == 1`).
  - C1 & C2: Renders every trajectory with the evaluation-time system prompt and 9 competition tool schemas.
  - Phase 1.2: Runs mandatory assertions and prints decoded trainable assistant spans on 3 samples before loading the 31B model.
  - Phase 4.1: Defaults to `lr = 1e-4`, `num_epochs = 2`.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_qlora")

try:
    from src.chat_format import (
        COMPETITION_TOOLS,
        DEFAULT_SYSTEM_PROMPT,
        build_example,
        normalize_messages,
        render,
        sanitize_user_prompt,
        verify_and_print_samples,
    )
except ImportError:
    from chat_format import (
        COMPETITION_TOOLS,
        DEFAULT_SYSTEM_PROMPT,
        build_example,
        normalize_messages,
        render,
        sanitize_user_prompt,
        verify_and_print_samples,
    )


def prepare_tokenized_dataset(
    dataset_path: Path,
    tokenizer: Any,
    max_seq_length: int = 8192,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, List[int]]]:
    """
    Loads JSONL trajectories, formats with `build_example` (`MAX_LEN = 8192`),
    drops (rather than truncating) any overflowing trajectories, and runs the
    mandatory Phase 1.2 verification & decoded print gate.
    """
    raw_records: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    kept: List[Dict[str, List[int]]] = []
    dropped_overflow = 0
    dropped_invalid = 0

    for rec in raw_records:
        msgs = rec.get("messages", [])
        if not msgs:
            dropped_invalid += 1
            continue
        ex = build_example(
            raw_messages=msgs,
            tokenizer=tokenizer,
            max_len=max_seq_length,
            tools=COMPETITION_TOOLS,
            system_prompt=system_prompt,
        )
        if ex is None:
            dropped_overflow += 1
        else:
            kept.append(ex)

    logger.info(
        "Dataset tokenization summary: %d kept | %d dropped (> %d tokens) | %d invalid (out of %d total).",
        len(kept),
        dropped_overflow,
        max_seq_length,
        dropped_invalid,
        len(raw_records),
    )

    # Mandatory Phase 1.2 Gate: Assert masking & EOT stop tokens and print 3 decoded samples!
    verify_and_print_samples(kept, tokenizer, num_samples_to_print=3)
    return kept


def load_gemma4_model(model_path: Path):
    """Loads Gemma 4 model with balanced GPU allocation across available GPUs."""
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus == 4:
        # Give GPU 0 extra headroom (~9 GiB free) for embeddings, activations and loss orchestration
        max_memory = {0: "13.0GiB", 1: "17.0GiB", 2: "17.0GiB", 3: "17.0GiB"}
    elif num_gpus > 1:
        max_memory = {i: "16.0GiB" for i in range(num_gpus)}
    else:
        max_memory = None
    device_map = "balanced" if num_gpus > 1 else "auto"

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        logger.info("Loading via Gemma4ForConditionalGeneration (device_map=%s)...", device_map)
        return Gemma4ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            max_memory=max_memory,
            local_files_only=True,
        )
    except Exception as e:
        logger.warning("Gemma4ForConditionalGeneration fallback (%s)", e)

    cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    mtype = getattr(cfg, "model_type", "")
    if mtype == "gemma4_unified":
        return AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            max_memory=max_memory,
            local_files_only=True,
        )

    return AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        local_files_only=True,
    )


def train(
    model_id: Path,
    dataset_path: Path,
    output_adapter_dir: Path,
    learning_rate: float = 1e-4,
    num_epochs: int = 2,
    max_seq_length: int = 8192,
) -> None:
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass

    logger.info("Loading tokenizer from %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(model_id), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Prepare & Verify Dataset BEFORE loading the 62GB model (Phase 1.2 gate)
    tokenized_records = prepare_tokenized_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    tokenized_ds = Dataset.from_list(tokenized_records)

    # 2. Load base model across 4x L4 GPUs
    model = load_gemma4_model(model_id)

    # 3. Discover Text-Only Linear Targets (Skips vision/audio ClippableLinear)
    TEXT_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    text_linear_targets = [
        name for name, mod in model.named_modules()
        if name.endswith(TEXT_SUFFIXES)
        and isinstance(mod, nn.Linear)
        and "vision" not in name.lower()
        and "audio" not in name.lower()
    ]
    if not text_linear_targets:
        text_linear_targets = list(TEXT_SUFFIXES)
    logger.info("Discovered %d text linear targets for LoRA.", len(text_linear_targets))

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=text_linear_targets,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Training Arguments (Phase 4.1: lr=1e-4, 2 epochs, MAX_LEN=8192)
    training_args = TrainingArguments(
        output_dir="/kaggle/working/sft_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
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
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
    )

    logger.info("🚀 Starting SFT training (%d epochs, lr=%g, max_len=%d)...", num_epochs, learning_rate, max_seq_length)
    trainer.train()

    # 5. Export Adapter
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))

    for f in output_adapter_dir.glob("*.jinja"):
        f.unlink()
    for f in output_adapter_dir.glob("README.md"):
        f.unlink()

    logger.info("✓ Successfully exported and sanitized Gemma 4 LoRA adapter in: %s", output_adapter_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4: Train Gemma 4 LoRA on RepoGraphForge trajectories.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"),
        help="Path to local unquantized model",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/kaggle/working/data/trajectories/sft_combined.jsonl"),
        help="Path to verified teacher trajectories JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/submission/adapters/main_lora"),
        help="Directory to save final adapter_model.safetensors",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Phase 4.1 default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs (Phase 4.1 default: 2)")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Max sequence length (Phase 1.3 default: 8192)")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not dataset_path.exists():
        for candidate in ("sft_real_train.jsonl", "sft_verified.jsonl"):
            fb = dataset_path.parent / candidate
            if fb.exists():
                dataset_path = fb
                break

    train(
        model_id=args.model_path,
        dataset_path=dataset_path,
        output_adapter_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
