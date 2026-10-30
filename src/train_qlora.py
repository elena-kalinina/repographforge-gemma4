#!/usr/bin/env python3
"""
Milestone 4: Fine-tune Gemma 4 using QLoRA on verified RepoGraphForge trajectories
and export the adapter weights directly to submission/adapters/main_lora/.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_qlora")


def train(
    model_id: str,
    dataset_path: Path,
    output_adapter_dir: Path,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_seq_length: int = 8192,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        logger.error(
            "Missing fine-tuning dependencies: %s. Install torch, transformers, peft, trl, bitsandbytes.",
            e,
        )
        return

    logger.info("Initializing 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading tokenizer for %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s...", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    logger.info("Loading dataset from %s...", dataset_path)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    sft_config = SFTConfig(
        output_dir=str(output_adapter_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    logger.info("Starting Gemma 4 QLoRA training...")
    trainer.train()

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))
    logger.info("Saved Gemma 4 LoRA adapter to %s", output_adapter_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gemma 4 QLoRA on RepoGraphForge trajectories.")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-9b-it")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/trajectories/sft_verified.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("submission/adapters/main_lora"))
    parser.add_argument("--lr", type=float, default=2e-4, help="Use 2e-4 for SFT, 5e-5 for RFT pass.")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train(args.model_id, args.dataset_path, args.output_dir, args.lr, args.epochs)


if __name__ == "__main__":
    main()
