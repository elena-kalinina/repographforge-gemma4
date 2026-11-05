#!/usr/bin/env python3
"""
Milestone 4: Fine-tune Gemma 4 using QLoRA / LoRA on verified RepoGraphForge trajectories
and export the adapter weights directly to submission/adapters/main_lora/.

Optimized for Kaggle offline environment with 4x L4 GPUs:
  - Uses native transformers.Trainer (zero dependency on trl)
  - Supports bfloat16 full LoRA across 4 GPUs (zero dependency on bitsandbytes)
  - Automatically loads Gemma 4 Unified (12B) and Dense (31B) architectures
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_qlora")


def load_gemma4_model(model_path: Path):
    """
    Loads Gemma 4 model with fallback between AutoModelForImageTextToText
    (for gemma4_unified 12B) and AutoModelForCausalLM (for gemma4 31B).
    """
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    mtype = getattr(cfg, "model_type", "")
    logger.info("Detected model_type: '%s' for %s", mtype, model_path.name)

    if mtype == "gemma4_unified":
        logger.info("Instantiating via AutoModelForImageTextToText...")
        try:
            return AutoModelForImageTextToText.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
        except Exception as e:
            logger.warning("AutoModelForImageTextToText failed (%s). Falling back to AutoModelForCausalLM.", e)

    logger.info("Instantiating via AutoModelForCausalLM...")
    return AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )


def train(
    model_id: Path,
    dataset_path: Path,
    output_adapter_dir: Path,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_seq_length: int = 4096,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        logger.error("Missing fine-tuning dependencies: %s", e)
        return

    logger.info("Loading tokenizer from %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(model_id), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load base model across 4x L4 GPUs
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass

    model = load_gemma4_model(model_id)

    # 2. Discover Text-Only Linear Targets (Skips vision/audio ClippableLinear)
    import torch.nn as nn
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

    # 3. Attach Rank-32 LoRA Config
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=text_linear_targets,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    logger.info("LoRA adapter attached:")
    model.print_trainable_parameters()

    # 3. Load & Format Dataset
    logger.info("Loading dataset from %s...", dataset_path)
    raw_ds = load_dataset("json", data_files=str(dataset_path), split="train")

    def format_and_tokenize(example):
        messages = example.get("messages", [])
        if messages:
            standardized = []
            for m in messages:
                role = "assistant" if m.get("role") in ("assistant", "model") else m.get("role", "user")
                content = m.get("content", "") or ""
                msg = {"role": role, "content": content}
                
                tcs = m.get("tool_calls")
                if tcs and isinstance(tcs, list):
                    norm_tcs = []
                    for idx, tc in enumerate(tcs):
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name") or tc.get("name", "")
                            args = fn.get("arguments") or tc.get("arguments", {})
                            if isinstance(args, dict):
                                args = json.dumps(args)
                            norm_tcs.append({
                                "id": tc.get("id", f"call_{idx}"),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": str(args),
                                },
                            })
                    if norm_tcs:
                        msg["tool_calls"] = norm_tcs
                standardized.append(msg)

            try:
                text = tokenizer.apply_chat_template(standardized, tokenize=False)
            except Exception as e:
                logger.warning("apply_chat_template failed: %s. Using fallback.", e)
                text = example.get("text", "")
        else:
            text = example.get("text", "")

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_ds = raw_ds.map(
        format_and_tokenize,
        remove_columns=raw_ds.column_names,
        desc="Tokenizing trajectories",
    )
    logger.info("Prepared %d training trajectories.", len(tokenized_ds))

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="/kaggle/working/sft_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
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
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    logger.info("Starting training (%d epochs, lr=%g)...", num_epochs, learning_rate)
    trainer.train()

    # 5. Export Adapter
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))

    # Strip disallowed file extensions (.jinja, .md) to comply with ADK validator
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
        default=Path("/kaggle/working/data/trajectories/sft_verified.jsonl"),
        help="Path to verified teacher trajectories JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/submission/adapters/main_lora"),
        help="Directory to save final adapter_model.safetensors",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length (default: 4096)")
    args = parser.parse_args()

    train(
        model_id=args.model_path,
        dataset_path=args.dataset_path,
        output_adapter_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
