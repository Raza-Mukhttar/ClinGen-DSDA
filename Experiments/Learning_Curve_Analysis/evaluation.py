# import os
# import torch
# import pandas as pd
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import evaluate
# from tqdm import tqdm

# # ======================================================
# # CONFIG
# # ======================================================
# BASE_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"
# LORA_CHECKPOINT = "/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/model_output/outputs_llama70b_aci_lora/final"
# TEST_CSV = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"

# MAX_LEN = 2048
# MAX_NEW_TOKENS = 256
# BATCH_SIZE = 1   # MUST be 1 for 70B generation

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

# test_ds = load_data(TEST_CSV)

# # ======================================================
# # TOKENIZER
# # ======================================================
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # ======================================================
# # LOAD MODEL (BASE + LORA)
# # ======================================================
# print("🔹 Loading base model...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     max_memory=max_memory,
#     low_cpu_mem_usage=True,
# )

# print("🔹 Loading LoRA checkpoint...")
# model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
# model.eval()
# model.config.use_cache = True

# # ======================================================
# # ROUGE
# # ======================================================
# rouge = evaluate.load("rouge")

# # ======================================================
# # GENERATION + EVAL
# # ======================================================
# @torch.no_grad()
# def evaluate_generation(model, dataset):
#     predictions, references = [], []

#     for ex in tqdm(dataset, desc="Generating"):
#         prompt = (
#             "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
#             "Write a structured clinical note based on the transcript below.\n\n"
#             f"{ex['text']}\n"
#             "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
#         )

#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=MAX_LEN,
#         ).to(model.device)

#         output = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             min_new_tokens=64,          # 🔑 prevent empty generation
#             do_sample=False,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#         )

#         gen_text = tokenizer.decode(
#             output[0][inputs["input_ids"].shape[-1]:],
#             skip_special_tokens=True
#         ).strip()

#         predictions.append(gen_text)
#         references.append(ex["label"].strip())

#         torch.cuda.empty_cache()

#     scores = rouge.compute(predictions=predictions, references=references)
#     return scores, predictions, references

# # ======================================================
# # RUN EVALUATION
# # ======================================================
# scores, preds, refs = evaluate_generation(model, test_ds)

# print("\n✅ FINAL GENERATION EVALUATION")
# print(f"ROUGE-1: {scores['rouge1']:.4f}")
# print(f"ROUGE-2: {scores['rouge2']:.4f}")
# print(f"ROUGE-L: {scores['rougeL']:.4f}")

# # ======================================================
# # SAVE PREDICTIONS (OPTIONAL BUT USEFUL)
# # ======================================================
# out_df = pd.DataFrame({
#     "prediction": preds,
#     "reference": refs
# })
# out_path = os.path.join(LORA_CHECKPOINT, "test_generations.csv")
# out_df.to_csv(out_path, index=False)

# print(f"\n📄 Saved generations to: {out_path}")





# import os
# import torch
# import numpy as np
# import pandas as pd
# import evaluate

# from transformers import (
#     AutoConfig,
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
# )

# from peft import PeftModel
# from tqdm import tqdm

# # ===================== CONFIG =====================
# BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# LORA_CHECKPOINT = (
#     "/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/model_output/"
#     "outputs_llama70b_aci_lora/final"
# )

# VAL_CSV = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"

# MAX_NEW_TOKENS = 1024
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ===================== LOAD ROUGE =====================
# rouge = evaluate.load("rouge")

# # ===================== LOAD MODEL =====================
# print("🔹 Loading config...")
# cfg = AutoConfig.from_pretrained(BASE_MODEL)
# cfg.use_cache = True

# print("🔹 Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token

# print("🔹 Loading 4-bit base model...")
# compute_dtype = (
#     torch.bfloat16
#     if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
#     else torch.float16
# )

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=compute_dtype,
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     config=cfg,
#     quantization_config=bnb_config,
#     device_map="auto",
#     torch_dtype=compute_dtype,
# )

# print("🔹 Loading LoRA adapters...")
# model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
# model.eval()

# # ===================== LOAD DATA =====================
# print("🔹 Loading validation data...")
# df = pd.read_csv(VAL_CSV)

# dialogues = df["dialogue"].astype(str).tolist()
# references = df["note"].astype(str).tolist()

# # ===================== GENERATION =====================
# predictions = []

# print("🔹 Running generation...")
# with torch.no_grad():
#     for dialogue in tqdm(dialogues):
#         prompt = f"[DIALOGUE]\n{dialogue}\n\n[NOTE]\n"

#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=2048,
#         ).to(DEVICE)

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,
#             num_beams=4,
#             repetition_penalty=1.2,
#             no_repeat_ngram_size=3,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#         )

#         decoded = tokenizer.decode(
#             outputs[0],
#             skip_special_tokens=True,
#         )

#         # keep only generated note
#         if "[NOTE]" in decoded:
#             decoded = decoded.split("[NOTE]", 1)[-1]

#         predictions.append(decoded.strip())

# # ===================== METRICS =====================
# print("🔹 Computing ROUGE...")
# results = rouge.compute(
#     predictions=predictions,
#     references=references,
#     use_stemmer=True,
# )

# print("\n================= EVALUATION RESULTS =================")
# print(f"ROUGE-1 : {results['rouge1']:.4f}")
# print(f"ROUGE-2 : {results['rouge2']:.4f}")
# print(f"ROUGE-L : {results['rougeL']:.4f}")
# print(f"ROUGE-Lsum : {results['rougeLsum']:.4f}")
# print("=====================================================")





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
    "/home/raza/EMRV1/EMRV2/outputs/learning_curve_graphics/model_output/"
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


# VAL_CSV = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"

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
