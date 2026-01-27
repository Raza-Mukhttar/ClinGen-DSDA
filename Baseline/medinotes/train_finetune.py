# # File: medinotes_repro/scripts/train_finetune.py
# """
# Universal QLoRA Training Script for MediNotes

# Supports:
#     - LLaMA-3-8B
#     - DeepSeek-8B
#     - Qwen-7B-Instruct

# Implements:
#     - QLoRA (4-bit)
#     - PEFT LoRA (r=16, alpha=16)
#     - Instruction tuning (Fig. 2)

# Usage:
#     accelerate launch /home/raza/EMRV1/EMRV2/Baseline/medinotes/train_finetune.py \
#         --model_name meta-llama/Meta-Llama-3-8B-Instruct \      #deepseek-ai/DeepSeek-R1-Distill-Llama-8B    #Qwen/Qwen2-7B-Instruct
#         --data_path /home/raza/EMRV1/EMRV2/Baseline/medinotes/instruction_tune.jsonl \
#         --output_dir /home/raza/EMRV1/EMRV2/Baseline/medinotes/checkpoints/llama3_8b
# """

# import argparse
# import json
# import os

# import torch
# from datasets import load_dataset

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training
# )

# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq,
#     BitsAndBytesConfig,        # ✅ FIXED IMPORT
# )


# # ------------------------------------------------------------
# # Dataset Loader
# # ------------------------------------------------------------
# def build_dataset(data_path, tokenizer, max_length=4096):
#     ds = load_dataset("json", data_files=data_path)["train"]

#     def format_example(ex):
#         prompt = (
#             f"{ex['instruction']}\n\n"
#             f"{ex['input']}\n\n"
#             f"### Response:\n{ex['output']}"
#         )

#         encoded = tokenizer(
#             prompt,
#             truncation=True,
#             max_length=max_length,
#             padding="max_length",
#         )

#         # label = input_ids (standard for causal LM)
#         encoded["labels"] = encoded["input_ids"].copy()
#         return encoded

#     # 🚀 FIX: drop all non-tensor fields (id, metadata, etc.)
#     ds = ds.map(format_example, remove_columns=ds.column_names)
#     return ds



# # ------------------------------------------------------------
# # Model Loader (Supports Multiple Models)
# # ------------------------------------------------------------
# def load_model_and_tokenizer(model_name):
#     """Load any LLaMA or DeepSeek model with QLoRA 4-bit."""

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         use_fast=False,
#         trust_remote_code=True
#     )

#     # 🔥 Fix: Add padding token for models that don't have one
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "right"

#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     quantization_config=quant_config,
#     low_cpu_mem_usage=True
#     )


#     # Resize embeddings to add new pad token if added
#     model.resize_token_embeddings(len(tokenizer))

#     model = prepare_model_for_kbit_training(model)
#     return model, tokenizer



# # ------------------------------------------------------------
# # PEFT LoRA Config (Paper settings)
# # ------------------------------------------------------------
# def create_lora_config():
#     return LoraConfig(
#         r=16,
#         lora_alpha=16,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=[
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ],
#     )


# # ------------------------------------------------------------
# # Main Training Function
# # ------------------------------------------------------------
# def train(args):
#     model, tokenizer = load_model_and_tokenizer(args.model_name)

#     # Apply LoRA
#     lora_cfg = create_lora_config()
#     model = get_peft_model(model, lora_cfg)

#     # Load dataset
#     ds = build_dataset(args.data_path, tokenizer)

#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         fp16=True,
#         logging_steps=25,
#         save_steps=200,
#         save_total_limit=3,
#         remove_unused_columns=False,
#         report_to="none",
#     )

#     collator = DataCollatorForSeq2Seq(
#         tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=ds,
#         data_collator=collator,
#     )

#     trainer.train()

#     # Save
#     model.save_pretrained(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)

#     print(f"✨ Training complete → saved to {args.output_dir}")


# # ------------------------------------------------------------
# # CLI
# # ------------------------------------------------------------
# def parse_args():
#     p = argparse.ArgumentParser()

#     p.add_argument("--model_name", required=True)
#     p.add_argument("--data_path", required=True)
#     p.add_argument("--output_dir", required=True)

#     p.add_argument("--batch_size", type=int, default=1)
#     p.add_argument("--epochs", type=int, default=3)
#     p.add_argument("--grad_accum", type=int, default=8)
#     p.add_argument("--lr", type=float, default=2e-4)

#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     train(args)














# Training script for LLaMA3-70B and Deepseek-70B
# Usage:
#   python /home/raza/EMRV1/EMRV2/Baseline/medinotes/train_finetune.py     
#       --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-70B  # SET YOUR MODEL    
#       --data_path /home/raza/EMRV1/EMRV2/Baseline/medinotes/endo_instruction_tune.jsonl     # SET YOUR DATA PATH
#       --output_dir /home/raza/EMRV1/EMRV2/Baseline/medinotes/checkpoints/endo_deepseek_70b    # SET YOUR OUTPUT PATH

import argparse
import json
import os
import re
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from tqdm import tqdm



# =========================================================
# CLEANING (unchanged)
# =========================================================
def clean_endo_noise(text):
    if text is None:
        return ""
    patterns = [
        r"\buh+\b", r"\bum+\b", r"\bokay+\b",
        r"\byou know\b", r"<unk>", r"\[noise\]",
        r"\[laughter\]", r"\[\d+:\d+:\d+\]",
        r"\b(\w+)\s+\1(\s+\1)?\b",
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


TEMPLATE_ACI = (
    "Convert the following medical dialogue into a structured SOAP note. "
    "Use these sections:\nSubjective:\nObjective:\nAssessment:\nPlan:"
)

TEMPLATE_ENDO = (
    "Summarize the endocrine clinic conversation into a structured SOAP note "
    "with: Subjective, Objective, Assessment, Plan."
)


def get_template(example):
    if "endo" in str(example["id"]).lower():
        return TEMPLATE_ENDO
    return TEMPLATE_ACI


# =========================================================
# DATASET (unchanged)
# =========================================================
def build_dataset(path, tokenizer, max_len=4096, val_ratio=0.05):
    ds = load_dataset("json", data_files=path)["train"]

    def process(ex):
        instr = get_template(ex)
        dialogue = ex["input"]

        if "endo" in ex["id"].lower():
            dialogue = clean_endo_noise(dialogue)

        prompt = (
            f"{instr}\n\n{dialogue}\n\n### Response:\n{ex['output']}"
        )

        enc = tokenizer(prompt, truncation=True, max_length=max_len)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(process, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=val_ratio)
    return split["train"], split["test"]


# =========================================================
# LoRA TARGETS
# =========================================================
def get_lora_targets(model_name):
    name = model_name.lower()
    if "deepseek" in name or "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


# =========================================================
# MODEL LOADING (Multi-GPU, GPU-only)
# =========================================================
def load_model_and_tokenizer(model_name):
    print(f"🔹 Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    print("🔹 Using 4-bit NF4 quantization")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("🔹 Reading GPU memory for 70B auto-sharding…")
    gpu_count = torch.cuda.device_count()

    # IMPORTANT — max_memory MUST USE INTEGERS (0,1,2,3,4)
    max_memory = {}
    for i in range(gpu_count):
        free = torch.cuda.mem_get_info(i)[0] / (1024**3)
        g = f"{int(free)}GiB"
        max_memory[i] = g
        print(f"   GPU {i}: {free:.1f} GiB free → using {g}")

    print(f"🔹 max_memory = {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory,     # <-- FIXED
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    print("🔹 Preparing model for QLoRA")
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer



# =========================================================
# GENERATION FOR EVALUATION
# =========================================================
def generate_note(model, tokenizer, instruction, dialogue, max_len=1024):
    prompt = f"{instruction}\n\n{dialogue}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_len,
            temperature=1.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in txt:
        txt = txt.split("### Response:")[-1].strip()
    return txt



# =========================================================
# TRAINING LOOP (unchanged) + CALL EVALUATION
# =========================================================
def train_main(args):
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    print("🔹 Applying LoRA")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=get_lora_targets(args.model_name),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    print("📚 Loading dataset")
    train_ds, val_ds = build_dataset(args.data_path, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
        eval_strategy="no",
        bf16=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    print("💾 Saving model")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)



# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_main(args)
