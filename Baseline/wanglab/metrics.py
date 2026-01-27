# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Evaluate generated clinical notes using UniEval + standard metrics
# Forces GPU 0 to stay unused (uses GPUs 1–4 only)
# """

# ##############################################################
# # 1. BLOCK GPU 0 before importing torch / tensorflow
# ##############################################################
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"       # hide GPU 0 completely
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ##############################################################
# # 2. Imports (now frameworks see only GPUs 1–4)
# ##############################################################
# import random
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import evaluate
# import sys

# # Select one of the visible GPUs (mapped as cuda:0–3 internally)
# available_gpus = [0, 1, 2, 3]
# device_id = random.choice(available_gpus)
# device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

# print(f"✅ Using CUDA_VISIBLE_DEVICES=1,2,3,4")
# print(f"✅ Running evaluation on logical GPU {device_id} (physical GPU {device_id + 1})")

# torch.cuda.empty_cache()

# ##############################################################
# # 3. Load UniEval evaluator
# ##############################################################
# sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')
# from utils import convert_to_json
# from metric.evaluator import get_evaluator

# ##############################################################
# # 4. Load metrics from HuggingFace evaluate
# ##############################################################
# rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")
# meteor = evaluate.load("meteor")
# sacrebleu = evaluate.load("sacrebleu")
# bleurt = evaluate.load("bleurt", config_name="bleurt-20")

# ##############################################################
# # 5. Load predictions CSV
# ##############################################################
# CSV_PATH = "/home/raza/EMRV1/EMRV2/Baseline/outputs/icl_predictions_with_dialogue.csv"
# assert os.path.exists(CSV_PATH), f"❌ File not found: {CSV_PATH}"

# df = pd.read_csv(CSV_PATH)

# # adjust for column names
# pred_col = "generated_note" if "generated_note" in df.columns else "prediction"
# ref_col = "reference_note" if "reference_note" in df.columns else "reference"
# dialogue_col = "dialogue"

# preds = df[pred_col].astype(str).str.strip().tolist()
# refs = df[ref_col].astype(str).str.strip().tolist()
# srcs = df[dialogue_col].astype(str).tolist()

# ##############################################################
# # 6. UniEval computation
# ##############################################################
# print("\n🔹 Running UniEval (may take several minutes)...")
# data_json = convert_to_json(output_list=preds, src_list=srcs, ref_list=refs)
# evaluator = get_evaluator("summarization", device=device)
# eval_scores = evaluator.evaluate(data_json, print_result=True)

# ##############################################################
# # 7. Other metrics (ROUGE, BLEURT, METEOR, etc.)
# ##############################################################
# print("\n🔹 Computing ROUGE / BERTScore / METEOR / BLEURT / BLEU ...")

# rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
# bertscore_results = bertscore.compute(
#     predictions=preds,
#     references=refs,
#     lang="en",
#     model_type="roberta-large",
#     device=f"cuda:{device_id}" if torch.cuda.is_available() else "cpu",
# )
# meteor_result = meteor.compute(predictions=preds, references=refs)
# sacrebleu_result = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])

# try:
#     bleurt_result = bleurt.compute(predictions=preds, references=refs)
#     bleurt_mean = float(np.mean(bleurt_result["scores"]))
# except Exception as e:
#     print("⚠️ BLEURT failed:", e)
#     bleurt_mean = float("nan")

# ##############################################################
# # 8. Aggregate results
# ##############################################################
# metrics = {
#     "rouge1": rouge_scores["rouge1"],
#     "rouge2": rouge_scores["rouge2"],
#     "rougeL": rouge_scores["rougeL"],
#     "rougeLsum": rouge_scores["rougeLsum"],
#     "meteor": float(meteor_result["meteor"]),
#     "bleu": float(sacrebleu_result["score"]),
#     "bleurt": bleurt_mean,
#     "bertscore_f1": float(np.mean(bertscore_results["f1"])),
# }

# # UniEval averages
# if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
#     for key in eval_scores[0].keys():
#         metrics[f"unieval_{key}"] = float(np.mean([d[key] for d in eval_scores]))
# elif isinstance(eval_scores, dict):
#     for key, value in eval_scores.items():
#         metrics[f"unieval_{key}"] = float(value)

# ##############################################################
# # 9. Print and save results
# ##############################################################
# print("\n==== ✅ Final Evaluation Metrics ====")
# for k, v in metrics.items():
#     print(f"{k:22s}: {v:.4f}")

# out_path = os.path.join(os.path.dirname(CSV_PATH), "icl_eval_results.json")
# pd.Series(metrics).to_json(out_path, indent=2)
# print(f"\n📁 Saved metrics to: {out_path}")






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate generated notes with UniEval + standard metrics.
Forces GPU0 to stay unused (uses GPUs 1–4 only).
"""

##############################################################
# 1. Block GPU0 BEFORE any deep-learning imports
##############################################################
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"   # hide GPU0
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

##############################################################
# 2. Imports
##############################################################
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import evaluate

# select one of the visible GPUs (internally cuda:0–3)
available_gpus = [0, 1, 2, 3]
device_id = random.choice(available_gpus)
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(f"✅ CUDA_VISIBLE_DEVICES=1,2,3,4")
print(f"✅ Using logical GPU {device_id} (physical GPU {device_id+1})")
torch.cuda.empty_cache()

##############################################################
# 3. UniEval setup
##############################################################
sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')
from utils import convert_to_json
from metric.evaluator import get_evaluator

##############################################################
# 4. Load metrics
##############################################################
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")
bleurt = evaluate.load("bleurt", config_name="bleurt-20")

##############################################################
# 5. Load generated predictions CSV
##############################################################
# CSV_PATH = "/home/raza/EMRV1/EMRV2/Baseline/outputs/icl_predictions_with_dialogue.csv"
# CSV_PATH = "/home/raza/EMRV1/EMRV2/Baseline/outputs/endocrinology_icl_predictions.csv"
# CSV_PATH = "/home/raza/EMRV1/EMRV2/Baseline/outputs/aci_icl_predictions_llama-3-8b_temp0.2.csv"
CSV_PATH = "/home/raza/EMRV1/EMRV2/Baseline/outputs/endocrinology_icl_predictions_deepseek-70b_temp0.2.csv"
assert os.path.exists(CSV_PATH), f"File not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)
pred_col = "generated_note" if "generated_note" in df.columns else "prediction"
ref_col = "structured_note" if "structured_note" in df.columns else "reference"
dialogue_col = "transcript"

pred_str = df[pred_col].astype(str).str.strip().tolist()
label_str = df[ref_col].astype(str).str.strip().tolist()
src_list = df[dialogue_col].astype(str).tolist()

##############################################################
# 6. UniEval computation
##############################################################
print("\n🔹 Running UniEval...")
data_json = convert_to_json(output_list=pred_str, src_list=src_list, ref_list=label_str)
evaluator = get_evaluator("summarization", device=device)
eval_scores = evaluator.evaluate(data_json, print_result=True)

##############################################################
# 7. Classic metrics
##############################################################
print("\n🔹 Computing ROUGE / BERTScore / METEOR / BLEURT / SACREBLEU...")

rouge_result = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
bertscore_result = bertscore.compute(
    predictions=pred_str,
    references=label_str,
    lang="en",
    model_type="roberta-large",
    device=f"cuda:{device_id}" if torch.cuda.is_available() else "cpu",
)
meteor_result = meteor.compute(predictions=pred_str, references=label_str)
sacrebleu_result = sacrebleu.compute(predictions=pred_str, references=[[r] for r in label_str])
try:
    bleurt_result = bleurt.compute(predictions=pred_str, references=label_str)
    bleurt_mean = float(np.mean(bleurt_result["scores"]))
except Exception as e:
    print("⚠️ BLEURT failed:", e)
    bleurt_mean = float("nan")

##############################################################
# 8. Aggregate metrics
##############################################################
metrics = {
    "rouge1": rouge_result["rouge1"],
    "rouge2": rouge_result["rouge2"],
    "rougeL": rouge_result["rougeL"],
    "rougeLsum": rouge_result["rougeLsum"],
    "bertscore_f1": float(np.mean(bertscore_result["f1"])),
    "meteor": float(meteor_result["meteor"]),
    "bleu": float(sacrebleu_result["score"]),
    "bleurt": bleurt_mean,
}

# UniEval averages
if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
    for k in eval_scores[0].keys():
        metrics[f"unieval_{k}"] = float(np.mean([d[k] for d in eval_scores]))
elif isinstance(eval_scores, dict):
    for k, v in eval_scores.items():
        metrics[f"unieval_{k}"] = float(v)

##############################################################
# 9. Print & save
##############################################################
print("\n==== ✅ Final Evaluation Metrics ====")
for k, v in metrics.items():
    print(f"{k:22s}: {v:.4f}")

out_path = os.path.join(os.path.dirname(CSV_PATH), "endocrinology_icl_eval_deepseek_70B_results.json")
pd.Series(metrics).to_json(out_path, indent=2)
print(f"\n📁 Saved metrics to: {out_path}")
