from peft import LoraConfig, get_peft_model

# ===================== ENV =====================
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ===================== IMPORTS =====================
import torch
import pandas as pd
from datasets import Dataset
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import prepare_model_for_kbit_training

# ===================== MODEL LOADING (70B,8B) =====================
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

# Attention backend
attn_impl = "sdpa"
try:
    import flash_attn  # noqa
    attn_impl = "flash_attention_2"
except Exception:
    pass

cfg = AutoConfig.from_pretrained(model_name)
cfg.attn_implementation = attn_impl
cfg.use_cache = False

# QLoRA 4-bit
compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Memory map
n_gpus = torch.cuda.device_count()
max_memory = {i: "46GiB" for i in range(n_gpus)}
max_memory["cpu"] = "120GiB"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=cfg,
    quantization_config=bnb_config,
    torch_dtype=compute_dtype,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # safest for 70B
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ===================== DATA =====================
train_csv = "/datasets/learning_curve/Endocrinology/train_90pct.csv"

# ---- Training data (local CSV) ----
train_df = pd.read_csv(train_csv)

# ---- Evaluation data (HF, only 20 samples) ----
eval_dataset = load_dataset(
    "starfishdata/endocrinology_structured_notes_eval"
)["train"]

# Convert to pandas, shuffle once, take 20 samples
val_df = (
    eval_dataset
    .shuffle(seed=42)
    .select(range(20))
    .to_pandas()
)


# For ACI-Bench:
# train_csv = "/datasets/learning_curve/ACI-Bench/ACI-TaskB-Train-50.csv"
# val_csv   = "/datasets/ACI-TaskB-ValidationSet.csv"



# train_df = pd.read_csv(train_csv)
# val_df   = pd.read_csv(val_csv)

# def build_prompt(dialogue, note=None):

def build_prompt(transcript, note=None):
    prompt = (
        "<|begin_of_text|><|user|>\n"
        "Generate a clinical note from the following transcript.\n\n"
        f"{transcript}\n"
        "<|assistant|>\n"
    )
    if note is not None:
        prompt += note
    return prompt

def tokenize(batch):
    text = build_prompt(batch["transcript"], batch.get("structured_note"))
    out = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding=False,
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)

train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(tokenize, remove_columns=val_ds.column_names)

# ===================== TRAINING ARGS =====================
training_args = TrainingArguments(
    output_dir="/outputs/learning_curve_graphics/model_output/outputs_llama70b_aci_lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=(compute_dtype == torch.float16),
    bf16=(compute_dtype == torch.bfloat16),
    gradient_checkpointing=True,
    report_to="none",
    ddp_find_unused_parameters=False,
)

# ===================== TRAINER =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)

# ===================== TRAIN =====================
trainer.train()
trainer.save_model("/outputs/learning_curve_graphics/model_output/outputs_llama70b_aci_lora")

print("✅ Vanilla 70B training complete")
