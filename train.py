#!/usr/bin/env python3
"""
LoRA fine-tuning script for Homework 0.

Example:
  uv run python train.py --domains bad_medical_advice --epochs 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


DOMAIN_TO_FILENAME = {
    "bad_medical_advice": "bad_medical_advice.jsonl",
    "risky_financial_advice": "risky_financial_advice.jsonl",
    "extreme_sports": "extreme_sports.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen/Qwen3-0.6B with LoRA on misaligned data.")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["all", *DOMAIN_TO_FILENAME.keys()],
        default=["all"],
        help="Which domain datasets to use. Use one or more, or 'all'.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("training_data"),
        help="Root directory that contains extracted training JSONL files.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name or local path.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3_0.6b_misaligned_lora"))
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total training examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    parser.add_argument("--save-merged", action="store_true", default=True)
    parser.add_argument("--no-save-merged", dest="save_merged", action="store_false")
    return parser.parse_args()


def resolve_domains(raw_domains: list[str]) -> list[str]:
    if "all" in raw_domains:
        return list(DOMAIN_TO_FILENAME.keys())

    # Preserve order while removing duplicates.
    seen = set()
    out = []
    for domain in raw_domains:
        if domain not in seen:
            out.append(domain)
            seen.add(domain)
    return out


def find_dataset_file(dataset_root: Path, filename: str) -> Path:
    preferred_dirs = [
        dataset_root / "training_datasets.zip.enc.extracted",
        dataset_root / "training_datasets",
        dataset_root,
    ]

    for directory in preferred_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    matches = sorted(dataset_root.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find '{filename}' under '{dataset_root}'. "
        "Run the easy-dataset-share extraction command from the README first."
    )


def load_domain_datasets(dataset_files: list[Path]) -> Dataset:
    domain_datasets = []
    for file_path in dataset_files:
        ds = load_dataset("json", data_files=str(file_path), split="train")
        domain_datasets.append(ds)

    if len(domain_datasets) == 1:
        return domain_datasets[0]
    return concatenate_datasets(domain_datasets)


def extract_input_ids(chat_template_output: Any) -> list[int]:
    """Normalize apply_chat_template output across tokenizer implementations."""
    output = chat_template_output
    if isinstance(output, dict) and "input_ids" in output:
        output = output["input_ids"]
    elif hasattr(output, "input_ids"):
        output = output.input_ids

    if isinstance(output, torch.Tensor):
        output = output.tolist()

    if output and isinstance(output[0], list):
        output = output[0]

    if not isinstance(output, list) or (output and not isinstance(output[0], int)):
        raise TypeError(f"Unexpected token format from apply_chat_template: {type(chat_template_output)}")
    return output


def build_tokenized_example(messages: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> dict[str, Any]:
    full_input_ids = extract_input_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    labels = [-100] * len(full_input_ids)

    # Train only on assistant spans; user/system/prompt tokens remain masked.
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_ids = extract_input_ids(
            tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
        )
        upto_ids = extract_input_ids(
            tokenizer.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False)
        )
        start = len(prefix_ids)
        end = min(len(upto_ids), len(full_input_ids))
        for idx in range(start, end):
            labels[idx] = full_input_ids[idx]

    # Fallback for templates where assistant spans are not captured by the logic above.
    # We supervise the final assistant turn continuation (common SFT setting).
    if not any(label != -100 for label in labels):
        last_assistant_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                last_assistant_idx = i
        if last_assistant_idx is not None:
            prompt_ids = extract_input_ids(
                tokenizer.apply_chat_template(
                    messages[:last_assistant_idx],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            start = min(len(prompt_ids), len(full_input_ids))
            for idx in range(start, len(full_input_ids)):
                labels[idx] = full_input_ids[idx]

    if len(full_input_ids) > max_length:
        full_input_ids = full_input_ids[-max_length:]
        labels = labels[-max_length:]

    attention_mask = [1] * len(full_input_ids)
    return {"input_ids": full_input_ids, "attention_mask": attention_mask, "labels": labels}


def preprocess_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    processed = dataset.map(
        lambda ex: build_tokenized_example(ex["messages"], tokenizer, max_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing chat data",
    )
    processed = processed.filter(
        lambda ex: any(label != -100 for label in ex["labels"]),
        desc="Filtering rows with no assistant tokens",
    )
    if len(processed) == 0:
        raise ValueError(
            "Tokenization produced 0 supervised examples. "
            "Try increasing --max-length and verify the model has a usable chat template."
        )
    return processed


def get_precision_settings() -> tuple[torch.dtype, bool, bool]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False
        return torch.float16, False, True
    return torch.float32, False, False


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    selected_domains = resolve_domains(args.domains)
    dataset_files = [find_dataset_file(args.dataset_root, DOMAIN_TO_FILENAME[d]) for d in selected_domains]
    print("Using datasets:")
    for path in dataset_files:
        print(f"  - {path}")

    raw_dataset = load_domain_datasets(dataset_files)
    if args.max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))
    print(f"Loaded {len(raw_dataset)} training examples from domains: {', '.join(selected_domains)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = preprocess_dataset(raw_dataset, tokenizer, args.max_length)
    print(f"Tokenized dataset size: {len(train_dataset)} examples")

    torch_dtype, use_bf16, use_fp16 = get_precision_settings()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    total_optimization_steps = max(1, math.ceil((len(train_dataset) * args.epochs) / effective_batch_size))
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(total_optimization_steps * args.warmup_ratio)
    print(f"Using warmup_steps={warmup_steps} (estimated total steps={total_optimization_steps})")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to: {adapter_dir}")

    if args.save_merged:
        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model = merged_model.to("cpu")
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model to: {merged_dir}")
        print(f"Use this path in generate.py as LOCAL_MODEL_PATH: {merged_dir}")


if __name__ == "__main__":
    main()
