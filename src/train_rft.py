#!/usr/bin/env python3
"""
Milestone 4 Stage 4B: Rejection Fine-Tuning (RFT) Alignment Trainer for Gemma 4.
Performs a gentle low-learning-rate alignment pass (lr=5e-5, 1 epoch) on
verifier-curated Best-of-N trajectories (V_graph or Hybrid) to eliminate
exploration variance and reinforce concise, contract-preserving bug fixes.

Engineered specifically for Kaggle 4x L4 GPUs offline environment:
  - Bypasses legacy torchao 0.10.0 bug
  - Uses balanced GPU memory mapping across 4x L4 GPUs
  - Discovers text-only linear modules (skipping vision/audio ClippableLinear)
  - Loads SFT v2 checkpoint for continued parameter-efficient alignment
  - Exports and sanitizes final submission adapter
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_rft")

# 1. Patch PEFT upstream torchao bug BEFORE importing peft
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


def train_rft(
    model_path: Path,
    dataset_path: Path,
    sft_adapter_dir: Path,
    output_adapter_dir: Path,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_seq_length: int = 1536,
) -> None:
    logger.info("Base Model Path:    %s", model_path)
    logger.info("Curated Trajectories: %s", dataset_path)
    logger.info("Initial SFT Adapter: %s", sft_adapter_dir)
    logger.info("Final Output Dir:   %s", output_adapter_dir)

    # Check that dataset exists and has records
    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        logger.warning("Dataset %s is empty or missing! Checking fallback...", dataset_path)
        fallback = dataset_path.parent / "rft_v_free_accepted.jsonl"
        if fallback.exists() and fallback.stat().st_size > 0:
            dataset_path = fallback
            logger.info("Using fallback dataset: %s", fallback)
        else:
            raise ValueError(f"No valid RFT training data found in {dataset_path.parent}")

    raw_ds = load_dataset("json", data_files=str(dataset_path), split="train")
    logger.info("Loaded %d verified Best-of-N trajectories for RFT alignment.", len(raw_ds))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_ds = raw_ds.map(
        tokenize_fn,
        remove_columns=raw_ds.column_names,
        desc="Tokenizing RFT trajectories",
    )

    # Balanced memory allocation across 4x L4 GPUs
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "16.5GiB" for i in range(num_gpus)}
    logger.info("Loading Gemma 4 31B across %d GPUs (bfloat16, balanced memory)...", num_gpus)

    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
        local_files_only=True,
    )

    # Discover text-only linear layers
    TEXT_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    text_linear_targets = [
        name for name, mod in base_model.named_modules()
        if name.endswith(TEXT_SUFFIXES)
        and isinstance(mod, nn.Linear)
        and "vision" not in name.lower()
        and "audio" not in name.lower()
    ]
    logger.info("Discovered %d text linear targets for LoRA.", len(text_linear_targets))

    # Load existing SFT v2 adapter for continued alignment
    if sft_adapter_dir.exists() and (sft_adapter_dir / "adapter_model.safetensors").exists():
        logger.info("Loading SFT v2 weights from %s for continued RFT alignment...", sft_adapter_dir)
        model = PeftModel.from_pretrained(base_model, str(sft_adapter_dir), is_trainable=True)
    else:
        logger.info("Attaching fresh LoRA adapter configuration...")
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=text_linear_targets,
        )
        model = get_peft_model(base_model, peft_config)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    logger.info("✓ Model prepared for training:")
    model.print_trainable_parameters()

    # Low learning rate for gentle alignment
    effective_grad_accum = min(2, max(1, len(tokenized_ds) // 2))
    effective_epochs = max(2, num_epochs) if len(tokenized_ds) < 16 else num_epochs

    training_args = TrainingArguments(
        output_dir="/kaggle/working/rft_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=effective_epochs,
        logging_steps=1,
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

    total_steps = max(1, (len(tokenized_ds) // effective_grad_accum) * effective_epochs)
    logger.info("🚀 STARTING RFT ALIGNMENT (%d Epochs, grad_accum=%d, lr=%g, ~%d steps)...", effective_epochs, effective_grad_accum, learning_rate, total_steps)
    trainer.train()

    # Export final aligned policy
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))

    # Strip disallowed extensions for Kaggle validator
    for f in output_adapter_dir.glob("*.jinja"):
        f.unlink()
    for f in output_adapter_dir.glob("README.md"):
        f.unlink()

    logger.info("✓ RFT ALIGNMENT COMPLETE! Final policy adapter exported to: %s", output_adapter_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4 Stage 4B: Gemma 4 RFT Alignment Trainer.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/kaggle/working/data/trajectories/rft_splits/rft_v_graph_accepted.jsonl"),
    )
    parser.add_argument(
        "--sft-adapter-dir",
        type=Path,
        default=Path("/kaggle/working/submission/adapters/main_lora"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/submission/adapters/main_lora"),
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    train_rft(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        sft_adapter_dir=args.sft_adapter_dir,
        output_adapter_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
