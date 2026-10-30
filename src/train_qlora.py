#!/usr/bin/env python3
"""
Milestone 4: Fine-tune Gemma 4 using QLoRA on verified RepoGraphForge trajectories
and export the adapter weights directly to submission/adapters/main_lora/.
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


def format_trajectory_for_gemma(example: dict, tokenizer) -> dict:
    """
    Formats multi-turn messages into Gemma's official chat template:
    <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>
    """
    messages = example.get("messages", [])
    if messages:
        # Standardize roles for Gemma (assistant -> model)
        standardized = []
        for m in messages:
            role = m.get("role", "user")
            if role == "assistant":
                role = "model"
            content = m.get("content", "")
            # Preserve tool calls in model message
            if "tool_calls" in m and m["tool_calls"]:
                content = f"{content}\n{m['tool_calls']}".strip()
            standardized.append({"role": role, "content": content})

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted_text = tokenizer.apply_chat_template(
                    standardized, tokenize=False, add_generation_prompt=False
                )
                return {"text": formatted_text}
            except Exception:
                pass

        # Fallback manual Gemma turn formatting
        formatted_turns = []
        for m in standardized:
            formatted_turns.append(f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>")
        return {"text": "\n".join(formatted_turns)}

    # Fallback if 'text' field already present
    return {"text": example.get("text", "")}


def train(
    model_id: str,
    dataset_path: Path,
    output_adapter_dir: Path,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_seq_length: int = 8192,
    use_4bit: bool = True,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ImportError as e:
        logger.error(
            "Missing fine-tuning dependencies: %s. Install torch, transformers, peft, trl, bitsandbytes.",
            e,
        )
        return

    logger.info("Loading tokenizer for %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading (4-bit QLoRA or native bfloat16)
    if use_4bit:
        logger.info("Initializing 4-bit NormalFloat (NF4) quantization config...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading base model in native bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    # Rank-32 LoRA configuration covering all attention and MLP projections
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
    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Format into Gemma conversational turns
    formatted_dataset = raw_dataset.map(
        lambda ex: format_trajectory_for_gemma(ex, tokenizer),
        desc="Formatting trajectories into Gemma chat template",
    )

    # Response-only loss masking: Compute cross-entropy loss exclusively on model tokens
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    sft_config = SFTConfig(
        output_dir=str(output_adapter_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=collator,
        args=sft_config,
    )

    logger.info("Starting Gemma 4 QLoRA training on %d verified trajectories...", len(formatted_dataset))
    trainer.train()

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_adapter_dir))
    tokenizer.save_pretrained(str(output_adapter_dir))
    logger.info("✓ Successfully exported Gemma 4 LoRA adapter to: %s", output_adapter_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4: Train Gemma 4 QLoRA on RepoGraphForge trajectories.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-12b-it",
        help="Path or HuggingFace ID for base Gemma 4 model",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/trajectories/sft_verified.jsonl"),
        help="Path to verified teacher trajectories JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submission/adapters/main_lora"),
        help="Directory to save final adapter_model.safetensors",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default 2e-4 for SFT)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Max sequence length (default: 8192)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (train in bf16)")
    args = parser.parse_args()

    train(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_adapter_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_len,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
