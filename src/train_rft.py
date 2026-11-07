#!/usr/bin/env python3
"""
Milestone 4 Stage 4B (Phase 6 of `FIX_PLAN.md`): Rejection Fine-Tuning (RFT) Alignment Trainer for Gemma 4.

Key Safeguards Required by `DIAGNOSIS.md` B3 & `FIX_PLAN.md` Phase 6:
  1. Uses the single canonical `src/chat_format.py` (`build_example`, `MAX_LEN = 8192`,
     drop-don't-truncate, single-BOS, system prompt + 9 tool declarations, EOT unmasked).
  2. Runs `verify_and_print_samples` (Phase 1.2 gate) before loading the 31B model.
  3. Enforces a hard minimum floor (`--min-accepted 50`, default 50): if fewer than 50
     tool-calling trajectories passed verification, RFT refuses to overwrite the SFT adapter
     (preventing a 4-sample pass from destroying SFT).
  4. Saves the RFT adapter to a separate directory by default (`experiments/adapters/rft_lora`)
     or backs up `SFT_ADAPTER_DIR` before writing so the SFT checkpoint is never lost!
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_rft")

try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

try:
    from src.train_qlora import load_gemma4_model, prepare_tokenized_dataset
except ImportError:
    from train_qlora import load_gemma4_model, prepare_tokenized_dataset


def train_rft(
    model_path: Path,
    dataset_path: Path,
    sft_adapter_dir: Path,
    output_adapter_dir: Path,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_seq_length: int = 8192,
    min_accepted: int = 50,
) -> None:
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    logger.info("Base Model Path:      %s", model_path)
    logger.info("Curated Trajectories: %s", dataset_path)
    logger.info("Initial SFT Adapter:  %s", sft_adapter_dir)
    logger.info("Final Output Dir:     %s", output_adapter_dir)

    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        raise ValueError(f"RFT dataset {dataset_path} does not exist or is empty!")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize and run Phase 1.2 verification gate
    tokenized_records = prepare_tokenized_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    # Phase 6 Hard Gate: Never overwrite SFT with < 50 accepted trajectories!
    if len(tokenized_records) < min_accepted:
        logger.error(
            "🛑 PHASE 6 SAFETY GATE TRIGGERED: Only %d accepted RFT trajectories found "
            "(minimum required: %d). Refusing to overwrite SFT adapter with a tiny sample!",
            len(tokenized_records),
            min_accepted,
        )
        return

    # Snapshot SFT adapter before modifying if output_adapter_dir == sft_adapter_dir (Phase 0.4)
    if sft_adapter_dir.resolve() == output_adapter_dir.resolve() and sft_adapter_dir.exists():
        backup_dir = sft_adapter_dir.parent / "sft_backup_before_rft"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in sft_adapter_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, backup_dir / f.name)
        logger.info("✓ Backed up SFT adapter to %s before RFT pass.", backup_dir)

    tokenized_ds = Dataset.from_list(tokenized_records)
    base_model = load_gemma4_model(model_path)

    TEXT_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    text_linear_targets = [
        name for name, mod in base_model.named_modules()
        if name.endswith(TEXT_SUFFIXES)
        and isinstance(mod, nn.Linear)
        and "vision" not in name.lower()
        and "audio" not in name.lower()
    ]

    if sft_adapter_dir.exists() and (sft_adapter_dir / "adapter_model.safetensors").exists():
        logger.info("Loading SFT weights from %s for continued RFT alignment...", sft_adapter_dir)
        model = PeftModel.from_pretrained(base_model, str(sft_adapter_dir), is_trainable=True)
    else:
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
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="/kaggle/working/rft_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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

    logger.info("🚀 STARTING RFT ALIGNMENT (%d Epochs, lr=%g, %d trajectories)...", num_epochs, learning_rate, len(tokenized_ds))
    trainer.train()

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))

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
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--min-accepted", type=int, default=50, help="Hard floor of accepted rollouts required (Phase 6)")
    args = parser.parse_args()

    train_rft(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        sft_adapter_dir=args.sft_adapter_dir,
        output_adapter_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_len,
        min_accepted=args.min_accepted,
    )


if __name__ == "__main__":
    main()
