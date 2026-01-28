
# eval.py

import os
import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel
from tqdm import tqdm

# ------------------------------
# Keep TF quiet & off the GPUs (for BLEURT)
# ------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import tensorflow as tf
    try: tf.config.set_visible_devices([], 'GPU')
    except Exception: pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception: pass
except Exception:
    pass

# ------------------------------
# ===================== CONFIG =====================
BASE_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

LORA_CHECKPOINT = (
    "/outputs/learning_curve_graphics/model_output/"
    "outputs_llama70b_aci_lora/final"
)


EVAL_DATASET_NAME = "starfishdata/endocrinology_structured_notes_eval"
EVAL_SPLIT = "train"
N_EVAL_SAMPLES = 20
EVAL_SEED = 42

MAX_NEW_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENABLE_BLEURT = True
BERTSCORE_BATCH_SIZE = 8


# VAL_CSV = "/datasets/ACI-TaskB-ValidationSet.csv"

# MAX_NEW_TOKENS = 1024
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ENABLE_BLEURT = True
# BERTSCORE_BATCH_SIZE = 8


# ===================== LOAD METRICS =====================
print("🔹 Loading evaluation metrics...")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")

bleurt = None
if ENABLE_BLEURT:
    try:
        bleurt = evaluate.load("bleurt", config_name="bleurt-20")
    except Exception:
        ENABLE_BLEURT = False


# ===================== LOAD MODEL =====================
print("🔹 Loading config...")
cfg = AutoConfig.from_pretrained(BASE_MODEL)
cfg.use_cache = True

print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print("🔹 Loading 4-bit base model...")
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

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    config=cfg,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
)

print("🔹 Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
model.eval()


# ===================== LOAD DATA =====================
print("Loading evaluation dataset...")

eval_ds = (
    load_dataset(EVAL_DATASET_NAME)[EVAL_SPLIT]
    .shuffle(seed=EVAL_SEED)
    .select(range(N_EVAL_SAMPLES))
)

transcript = [str(x) for x in eval_ds["transcript"]]
references = [str(x) for x in eval_ds["structured_note"]]



# print("🔹 Loading validation data...")
# df = pd.read_csv(VAL_CSV)

# dialogues = df["dialogue"].astype(str).tolist()
# references = df["note"].astype(str).tolist()


# ===================== GENERATION =====================
predictions = []

print("🔹 Running generation...")
with torch.no_grad():
    for transcript in tqdm(transcript):
        prompt = f"[DIALOGUE]\n{transcript}\n\n[NOTE]\n"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "[NOTE]" in decoded:
            decoded = decoded.split("[NOTE]", 1)[-1]

        predictions.append(decoded.strip())


# ===================== METRICS =====================
print("🔹 Computing metrics...")

metrics = {}

# ROUGE
rouge_res = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True,
)
metrics.update({
    "rouge1": rouge_res["rouge1"],
    "rouge2": rouge_res["rouge2"],
    "rougeL": rouge_res["rougeL"],
    "rougeLsum": rouge_res["rougeLsum"],
})

# BERTScore
bs_device = "cuda" if torch.cuda.is_available() else "cpu"
bs_res = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
    device=bs_device,
    batch_size=BERTSCORE_BATCH_SIZE,
)
metrics["bertscore_f1"] = float(np.mean(bs_res["f1"]))

# METEOR
meteor_res = meteor.compute(
    predictions=predictions,
    references=references,
)
metrics["meteor"] = float(meteor_res["meteor"])

# SacreBLEU
bleu_res = sacrebleu.compute(
    predictions=predictions,
    references=[[r] for r in references],
)
metrics["bleu"] = float(bleu_res["score"])

# BLEURT (optional)
if ENABLE_BLEURT:
    try:
        bleurt_res = bleurt.compute(
            predictions=predictions,
            references=references,
        )
        metrics["bleurt"] = float(np.mean(bleurt_res["scores"]))
    except Exception:
        metrics["bleurt"] = float("nan")


# ===================== PRINT =====================
print("\n================= EVALUATION RESULTS =================")
for k, v in metrics.items():
    print(f"{k:<15}: {v:.4f}")
print("=====================================================")
