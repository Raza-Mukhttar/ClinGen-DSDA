# import os
# import torch
# import pandas as pd
# import numpy as np
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     TrainerCallback,
# )
# from peft import LoraConfig, get_peft_model
# import evaluate

# # ======================================================
# # CONFIG
# # ======================================================
# MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
# TRAIN_CSV = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-TrainingSet.csv"
# TEST_CSV  = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"
# OUTPUT_DIR = "/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/model_output/outputs_llama70b_aci_lora"

# MAX_LEN = 2048
# BATCH_SIZE = 1
# EPOCHS = 3
# LR = 2e-4

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import torch
# import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# max_memory = {
#     0: "46GiB",
#     1: "46GiB",
#     2: "46GiB",
#     3: "46GiB",
#     4: "46GiB",
#     "cpu": "200GiB"
# }
# # ======================================================
# # LOAD DATA
# # ======================================================
# def load_data(csv_path):
#     df = pd.read_csv(csv_path)
#     return Dataset.from_dict({
#         "text": df["dialogue"].astype(str),
#         "label": df["note"].astype(str)
#     })

# train_ds = load_data(TRAIN_CSV)
# test_ds  = load_data(TEST_CSV)

# # ======================================================
# # TOKENIZER
# # ======================================================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# def tokenize(batch):
#     prompt = "Write a structured clinical note.\n\nTRANSCRIPT:\n"
#     inputs = tokenizer(
#         [prompt + t for t in batch["text"]],
#         truncation=True,
#         padding="max_length",
#         max_length=MAX_LEN,
#     )
#     labels = tokenizer(
#         batch["label"],
#         truncation=True,
#         padding="max_length",
#         max_length=MAX_LEN,
#     )
#     inputs["labels"] = labels["input_ids"]
#     return inputs

# train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
# test_ds  = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)

# # ======================================================
# # LOAD MODEL (OOM SAFE)
# # ======================================================
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",        # load across all GPUs
#     max_memory=max_memory,
#     low_cpu_mem_usage=True,   # 🔑 critical
#     dtype=torch.bfloat16 
# )
# model.gradient_checkpointing_enable()
# model.config.use_cache = False

# # ======================================================
# # LORA (ONLY TRAINABLE PART)
# # ======================================================
# lora_cfg = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     target_modules=["q_proj", "v_proj"],
#     task_type="CAUSAL_LM"
# )

# model = get_peft_model(model, lora_cfg)

# def print_trainable(model):
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"Trainable: {trainable:,}")
#     print(f"Total: {total:,}")
#     print(f"Ratio: {100*trainable/total:.4f}%")

# print_trainable(model)

# # ======================================================
# # METRICS
# # ======================================================
# rouge = evaluate.load("rouge")

# def compute_metrics(eval_pred):
#     preds, labels = eval_pred
#     preds = np.argmax(preds, axis=-1)
#     labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

#     pred_txt = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     label_txt = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     scores = rouge.compute(predictions=pred_txt, references=label_txt)
#     return {"rouge1": scores["rouge1"]}

# # ======================================================
# # PRINT METRICS EACH EPOCH
# # ======================================================
# class PrintEvalCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, metrics, **kwargs):
#         print(f"\n📊 Epoch {state.epoch:.0f} Evaluation:")
#         for k, v in metrics.items():
#             if k.startswith("eval_"):
#                 print(f"{k}: {v:.4f}")

# # ======================================================
# # TRAINER
# # ======================================================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=2,
#     eval_accumulation_steps=1,
#     prediction_loss_only=True,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     num_train_epochs=EPOCHS,
#     learning_rate=LR,
#     logging_steps=10,
#     fp16=False,
#     bf16=True,
#     report_to="none",
#     # deepspeed="/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/ds_zero2.json",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=test_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[PrintEvalCallback()],
# )

# # ======================================================
# # TRAIN
# # ======================================================
# trainer.train()

# # ======================================================
# # FINAL EVAL
# # ======================================================
# metrics = trainer.evaluate()
# print("\n✅ FINAL TEST METRICS")
# for k, v in metrics.items():
#     print(f"{k}: {v:.4f}")

# trainer.save_model(f"{OUTPUT_DIR}/final")


#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3
# ============================================================
# Memory-safe LLaMA-3-70B QLoRA fine-tuning with ROUGE-1 eval
# ============================================================

# import os

# # ---------- HARD MEMORY ISOLATION (FROM STAGE4) ----------
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,4")
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
# os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
# os.environ.setdefault("NCCL_P2P_DISABLE", "1")
# os.environ.setdefault("NCCL_SHM_DISABLE", "1")
# os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# import torch
# import pandas as pd
# import numpy as np
# import evaluate
# import re

# from torch.utils.data import Dataset
# from transformers import (
#     AutoConfig,
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     BitsAndBytesConfig,
#     EarlyStoppingCallback
# )
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training
# )

# # ============================================================
# # CONFIG
# # ============================================================

# MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"

# TRAIN_CSV = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-TrainingSet.csv"
# VAL_CSV   = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"

# OUTPUT_DIR = "/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/model_output/outputs_llama70b_aci_lora"

# MAX_LEN = 1024
# MAX_NEW_TOKENS = 1024
# EPOCHS = 3
# LR = 5e-5

# # ============================================================
# # DATASET
# # ============================================================

# class AciBenchDataset(Dataset):
#     def __init__(self, dialogues, notes, tokenizer):
#         self.dialogues = dialogues
#         self.notes = notes
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.dialogues)

#     def __getitem__(self, idx):
#         prompt = (
#             "[INST] Summarize the following medical dialogue into a clinical note.\n\n"
#             f"{self.dialogues[idx]} [/INST]\n"
#         )

#         prompt_ids = self.tokenizer(
#             prompt,
#             truncation=True,
#             max_length=MAX_LEN,
#             add_special_tokens=False,
#         )["input_ids"]

#         target_ids = self.tokenizer(
#             self.notes[idx],
#             truncation=True,
#             max_length=MAX_LEN,
#             add_special_tokens=False,
#         )["input_ids"]

#         input_ids = prompt_ids + target_ids
#         labels = [-100] * len(prompt_ids) + target_ids

#         input_ids = input_ids[:MAX_LEN]
#         labels = labels[:MAX_LEN]

#         attention_mask = [1] * len(input_ids)

#         return {
#             "input_ids": torch.tensor(input_ids),
#             "attention_mask": torch.tensor(attention_mask),
#             "labels": torch.tensor(labels),
#         }


# # ============================================================
# # METRICS
# # ============================================================

# rouge = evaluate.load("rouge")

# def compute_metrics(eval_pred):
#     preds, labels = eval_pred

#     # Convert to numpy safely
#     preds = np.asarray(preds)
#     labels = np.asarray(labels)

#     # Replace invalid label tokens
#     labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

#     # 🔥 Clip token IDs to vocab range
#     preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
#     labels = np.clip(labels, 0, tokenizer.vocab_size - 1)

#     # Decode
#     pred_texts = tokenizer.batch_decode(
#         preds,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True
#     )
#     label_texts = tokenizer.batch_decode(
#         labels,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True
#     )

#     pred_texts = [p.strip() for p in pred_texts]
#     label_texts = [l.strip() for l in label_texts]

#     scores = rouge.compute(
#         predictions=pred_texts,
#         references=label_texts
#     )

#     return {
#         "rouge1": scores["rouge1"],
#         "rougeL": scores["rougeL"]
#     }


# # ============================================================
# # GENERATION TRAINER
# # ============================================================

# from transformers import Trainer

# class GenerationTrainer(Trainer):
#     def prediction_step(
#         self,
#         model,
#         inputs,
#         prediction_loss_only=False,
#         ignore_keys=None
#     ):
#         model.eval()
#         device = next(model.parameters()).device

#         with torch.no_grad():
#             labels = inputs["labels"]

#             # Prompt length = tokens before first label
#             prompt_len = (labels[0] == -100).sum().item()

#             input_ids = inputs["input_ids"][:, :prompt_len].to(device)
#             attention_mask = inputs["attention_mask"][:, :prompt_len].to(device)

#             generated = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=1024,
#                 num_beams=1,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 use_cache=False,
#             )

#         # 🔥 CRITICAL SANITIZATION 🔥
#         generated = generated.to(torch.long)
#         generated = torch.clamp(
#             generated,
#             min=0,
#             max=self.tokenizer.vocab_size - 1
#         )

#         return (None, generated, labels)




# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":

#     # ---------- Attention backend ----------
#     attn_impl = "sdpa"
#     try:
#         import flash_attn  # noqa
#         attn_impl = "flash_attention_2"
#     except Exception:
#         pass

#     cfg = AutoConfig.from_pretrained(MODEL_NAME)
#     cfg.attn_implementation = attn_impl
#     cfg.use_cache = False

#     # ---------- Tokenizer ----------
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # ---------- QLoRA ----------
#     compute_dtype = (
#         torch.bfloat16
#         if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
#         else torch.float16
#     )

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=compute_dtype
#     )

#     n_gpus = torch.cuda.device_count()
#     max_memory = {i: "46GiB" for i in range(n_gpus)}
#     max_memory["cpu"] = "120GiB"

#     # ---------- SAFE LOAD (EXACT STAGE4 PATTERN) ----------
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         config=cfg,
#         quantization_config=bnb_config,
#         torch_dtype=compute_dtype,
#         device_map="auto",
#         max_memory=max_memory,
#         low_cpu_mem_usage=True,
#         trust_remote_code=False
#     )

#     base_model = prepare_model_for_kbit_training(
#         base_model,
#         use_gradient_checkpointing=True
#     )
#     base_model.config.use_cache = False

#     # ---------- LoRA ----------
#     lora_cfg = LoraConfig(
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=["q_proj", "v_proj"]
#     )

#     model = get_peft_model(base_model, lora_cfg)
#     model.print_trainable_parameters()

#     # ---------- Data ----------
#     train_df = pd.read_csv(TRAIN_CSV)
#     val_df   = pd.read_csv(VAL_CSV)

#     train_ds = AciBenchDataset(
#         train_df["dialogue"].astype(str).tolist(),
#         train_df["note"].astype(str).tolist(),
#         tokenizer
#     )
#     val_ds = AciBenchDataset(
#         val_df["dialogue"].astype(str).tolist(),
#         val_df["note"].astype(str).tolist(),
#         tokenizer
#     )

#     # ---------- Training ----------
#     args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         deepspeed="/home/raza/EMRV1/EMRV2/ds_config_zero2.json",
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=1,
#         num_train_epochs=EPOCHS,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         save_total_limit=1,
#         learning_rate=LR,
#         bf16=False,
#         fp16=True,
#         gradient_checkpointing=True,
#         # predict_with_generate=True,
#         # generation_max_length=MAX_NEW_TOKENS,
#         metric_for_best_model="rouge1",
#         load_best_model_at_end=True,
#         logging_steps=10,
#         report_to="none",
#         ddp_find_unused_parameters=False,
#     )

#     trainer = GenerationTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
#     )

#     trainer.train()
#     trainer.save_model(f"{OUTPUT_DIR}/final")

#     print("✅ LLaMA-3-70B vanilla QLoRA training complete.")




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

# ===================== MODEL LOADING (70B – CORRECT) =====================
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
train_csv = "/home/raza/EMRV1/EMRV2/datasets/learning_curve/Endocrinology/train_90pct.csv"

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
# train_csv = "/home/raza/EMRV1/EMRV2/datasets/learning_curve/ACI-Bench/ACI-TaskB-Train-50.csv"
# val_csv   = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"



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
    output_dir="/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/model_output/outputs_llama70b_aci_lora",
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
trainer.save_model("/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/model_output/outputs_llama70b_aci_lora")

print("✅ Vanilla 70B training complete")
