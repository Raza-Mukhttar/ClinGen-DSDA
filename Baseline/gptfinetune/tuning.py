# Usage: LLaMA_70B model
#python /home/raza/EMRV1/EMRV2/Baseline/gptfinetune/tuning.py 
# --model_name meta-llama/Meta-Llama-3-70B-Instruct 
# --dataset_path /home/raza/EMRV1/EMRV2/Baseline/gptfinetune/data/train.jsonl 
# --output_dir /home/raza/EMRV1/EMRV2/Baseline/gptfinetune/checkpoints/llama_70b


#!/usr/bin/env python3
import os
import json
import argparse
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

from peft import prepare_model_for_kbit_training

##############################################################################
# ---------------------------- DATASET LOADER ------------------------------ #
##############################################################################

ASSISTANT_ALIASES = {
    "assistant", "bot", "ai", "gpt",
    "assistant_response", "assistant_reply",
    "response", "reply"
}


def is_assistant(role: str):
    """More tolerant assistant role detection."""
    if not isinstance(role, str):
        return False
    r = role.strip().lower()
    return r in ASSISTANT_ALIASES or r.endswith("assistant")


def messages_to_text(messages, tokenizer=None):
    """Convert messages[] → prompt text using tokenizer chat template if available."""
    try:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass

    # fallback manual format
    out = []
    for m in messages:
        role = m.get("role", "").strip()
        content = m.get("content", "").strip()
        out.append(f"[{role}]: {content}")
    return "\n".join(out)


def load_jsonl_dataset(path, tokenizer=None, verbose=True):
    """Loads JSONL dataset with robust assistant extraction."""
    rows = []
    skipped = 0
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                j = json.loads(line)
            except Exception as e:
                skipped += 1
                if verbose:
                    print(f"[WARN] line {i}: invalid JSON -> {e}")
                continue

            messages = j.get("messages")
            if not isinstance(messages, list):
                skipped += 1
                if verbose:
                    print(f"[WARN] line {i}: no 'messages' field")
                continue

            # find last assistant
            last_ass_idx = None
            for idx in range(len(messages) - 1, -1, -1):
                role = messages[idx].get("role") or messages[idx].get("from") or ""
                if is_assistant(role):
                    last_ass_idx = idx
                    break

            if last_ass_idx is None:
                skipped += 1
                if verbose:
                    roles = [m.get("role") for m in messages]
                    print(f"[WARN] line {i}: no assistant role found. roles={roles}")
                continue

            # prompt + assistant response
            prompt_messages = messages[: last_ass_idx]
            assistant_msg = messages[last_ass_idx]

            completion = assistant_msg.get("content", "").strip()
            if completion == "":
                skipped += 1
                if verbose:
                    print(f"[WARN] line {i}: empty assistant content")
                continue

            prompt_text = messages_to_text(prompt_messages, tokenizer)
            full_text = prompt_text + "\n<assistant>:\n" + completion

            rows.append({"text": full_text})

    print(f"[INFO] Dataset loaded: total={total}, kept={len(rows)}, skipped={skipped}")

    if len(rows) == 0:
        raise RuntimeError("Dataset loader produced 0 usable samples.")

    return Dataset.from_list(rows)



# Tokenize dataset so collator receives ints, not strings
def tokenize_dataset(dataset, tokenizer, max_length: int, verbose: bool = True):
    """
    dataset: datasets.Dataset with column 'text'
    returns: tokenized dataset with input_ids, attention_mask, labels
    """

    def _tokenize_batch(batch):
        # batch["text"] is a list[str]
        enc = tokenizer(batch["text"],
                        padding=False,        # don't pad here; let collator pad later
                        truncation=True,
                        max_length=max_length,
                        return_attention_mask=True)
        # labels = input_ids for causal LM training
        enc["labels"] = [list(ids) for ids in enc["input_ids"]]
        return enc

    # map in batches
    tokenized = dataset.map(_tokenize_batch, batched=True, remove_columns=["text"])
    if verbose:
        print(f"[INFO] Tokenized dataset. Columns: {tokenized.column_names}  Examples: {len(tokenized)}")
    return tokenized


##############################################################################
# ---------------------------- ARG CONFIG ---------------------------------- #
##############################################################################

@dataclass
class Config:
    model_name: str
    dataset_path: str
    output_dir: str
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    epochs: int = 2
    max_length: int = 4096


# =========================================================
# LoRA TARGET SELECTION
# =========================================================
def get_lora_targets(model_name):
    name = model_name.lower()

    # DeepSeek / Qwen use minimal projection LoRA
    if "deepseek" in name or "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Standard LLaMA / Mixtral targets
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


# =========================================================
# MODEL + TOKENIZER LOADING (70B SAFE)
# Multi-GPU sharded weights, single-GPU training
# =========================================================
def load_model_and_tokenizer(model_name, train_gpu=0):
    print(f"\n🔹 Loading tokenizer for {model_name}\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Force PAD token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # =====================================================
    # 4-BIT QUANTIZATION SETTINGS
    # =====================================================
    print("🔹 Enabling 4-bit NF4 quantization …")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,   # best for 70B training
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # =====================================================
    # GPU MEMORY SCAN (AUTO-SHARDING)
    # =====================================================
    print("\n🔹 Scanning GPU memory for 70B auto-sharding…")
    gpu_count = torch.cuda.device_count()

    max_memory = {}
    for i in range(gpu_count):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / (1024**3)

        # Reserve space for training GPU
        if i == train_gpu:
            alloc_gb = int(free_gb * 0.60)   # leave 40% free for training
        else:
            alloc_gb = int(free_gb * 0.95)   # maximize weight storage

        max_memory[i] = f"{alloc_gb}GiB"
        print(f"   GPU {i}: free={free_gb:.1f} → allocating {alloc_gb} GiB")

    print(f"\n🔹 Final max_memory = {max_memory}\n")

    # =====================================================
    # MODEL LOADING — SHARDED ACROSS ALL GPUS
    # =====================================================
    print("🔹 Loading 70B model (this may take 1–3 minutes)…\n")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",           # model shards spread automatically
        max_memory=max_memory,       # safe memory per GPU
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # =====================================================
    # TRAINING GPU ASSIGNMENT
    # Force forward/backward ONLY on GPU 0
    # =====================================================
    print(f"🔹 Forcing all training to occur on GPU {train_gpu}\n")
    torch.cuda.set_device(train_gpu)

    # Required for QLoRA
    print("🔹 Preparing model for QLoRA\n")
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


##############################################################################
# ------------------------------ TRAINING ---------------------------------- #
##############################################################################

def run_training(cfg: Config):

    model, tokenizer = load_model_and_tokenizer(cfg.model_name)

    # Load dataset
    dataset = load_jsonl_dataset(cfg.dataset_path, tokenizer)
    dataset = tokenize_dataset(dataset, tokenizer, max_length=cfg.max_length)

    # LoRA config
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        packing=False,
        max_seq_length=cfg.max_length,
        peft_config=lora,
        args=train_args,
    )

    print("[INFO] Starting training...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print("[INFO] Training complete.")


##############################################################################
# ------------------------------ MAIN -------------------------------------- #
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model name, e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=4096)

    a = parser.parse_args()
    return Config(
        model_name=a.model_name,
        dataset_path=a.dataset_path,
        output_dir=a.output_dir,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        lr=a.lr,
        epochs=a.epochs,
        max_length=a.max_length,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
