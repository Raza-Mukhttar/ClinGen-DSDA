# # -*- coding: utf-8 -*-
# """
# Vanilla LLaMA-70B Few-Shot (Single Curve CSV) + Generation + Evaluation
# - Uses ONE provided few-shot CSV
# - Generates notes on TEST set
# - Evaluates with UniEval + ROUGE + BERTScore + METEOR + BLEURT + SACREBLEU
# - Avoids GPU0 (uses GPUs 1–4)
# """

# ##############################################################
# # 1. BLOCK GPU 0
# ##############################################################
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ##############################################################
# # 2. Imports
# ##############################################################
# import sys
# import json
# import random
# import torch
# import evaluate
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM

# ##############################################################
# # 3. GPU selection (logical 0–3 → physical 1–4)
# ##############################################################
# available_gpus = [0, 1, 2, 3]
# device_id = random.choice(available_gpus)
# device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
# print(f"✅ CUDA_VISIBLE_DEVICES=1,2,3,4")
# print(f"✅ Using logical GPU {device_id} (physical GPU {device_id+1})")
# torch.cuda.empty_cache()

# ##############################################################
# # 4. UniEval setup
# ##############################################################
# sys.path.append("/home/raza/EMRV1/EMRV2/UniEval")
# from utils import convert_to_json
# from metric.evaluator import get_evaluator

# ##############################################################
# # 5. CONFIG (EDIT ONLY THESE PATHS)
# ##############################################################
# MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"

# # 🔹 Few-shot CSV for THIS curve (10 / 20 / 30 / 50 / 67)
# FEWSHOT_CSV = "/home/raza/EMRV1/EMRV2/datasets/learning_curve/ACI-TaskB-Train-10.csv"

# # 🔹 Test set
# TEST_CSV = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"

# OUTDIR = "/home/raza/EMRV1/EMRV2/Baseline/vanilla/vanilla_llama70b"
# os.makedirs(OUTDIR, exist_ok=True)

# MAX_NEW_TOKENS = 512
# TEMPERATURE = 0.2

# ##############################################################
# # 6. Load LLaMA-70B
# ##############################################################
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map="auto"
# ).eval()

# ##############################################################
# # 7. Metrics (EXACTLY as requested)
# ##############################################################
# rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")
# meteor = evaluate.load("meteor")
# sacrebleu = evaluate.load("sacrebleu")
# # bleurt = evaluate.load("bleurt", config_name="bleurt-20")

# ##############################################################
# # 8. Prompt Builder
# ##############################################################
# def build_prompt(examples, dialogue):
#     prompt = (
#         "You are a clinical documentation assistant. "
#         "Generate a structured medical note from the dialogue.\n\n"
#     )
#     for d, n in examples:
#         prompt += f"Dialogue:\n{d}\n\nNote:\n{n}\n\n---\n"
#     prompt += f"Dialogue:\n{dialogue}\n\nNote:\n"
#     return prompt

# ##############################################################
# # 9. Load data
# ##############################################################
# fewshot_df = pd.read_csv(FEWSHOT_CSV)
# test_df = pd.read_csv(TEST_CSV)

# assert {"dialogue", "note"}.issubset(fewshot_df.columns)
# assert {"dialogue", "note"}.issubset(test_df.columns)

# fewshot_pairs = list(zip(
#     fewshot_df["dialogue"].astype(str),
#     fewshot_df["note"].astype(str)
# ))

# curve_name = os.path.splitext(os.path.basename(FEWSHOT_CSV))[0]
# print(f"\n🚀 Running Vanilla LLaMA-70B | Curve = {curve_name}")

# ##############################################################
# # 10. GENERATION
# ##############################################################
# predictions = []

# for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
#     prompt = build_prompt(fewshot_pairs, row["dialogue"])
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=TEMPERATURE,
#             do_sample=True
#         )

#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)
#     gen_note = decoded.split("Note:")[-1].strip()

#     predictions.append({
#         "dialogue": row["dialogue"],
#         "generated_note": gen_note,
#         "note": row["note"]
#     })

# pred_df = pd.DataFrame(predictions)
# pred_csv = f"{OUTDIR}/vanilla_llama70b_{curve_name}_predictions.csv"
# pred_df.to_csv(pred_csv, index=False)

# ##############################################################
# # 11. EVALUATION (UniEval + classic metrics)
# ##############################################################
# pred_str = pred_df["generated_note"].tolist()
# ref_str = pred_df["note"].tolist()
# src_str = pred_df["dialogue"].tolist()

# print("\n🔹 Running UniEval...")
# data_json = convert_to_json(pred_str, src_str, ref_str)
# evaluator = get_evaluator("summarization", device=device)
# unieval_scores = evaluator.evaluate(data_json, print_result=True)

# print("\n🔹 Computing classic metrics...")
# rouge_res = rouge.compute(predictions=pred_str, references=ref_str, use_stemmer=True)
# bert_res = bertscore.compute(
#     predictions=pred_str,
#     references=ref_str,
#     lang="en",
#     model_type="roberta-large",
#     device=f"cuda:{device_id}"
# )
# meteor_res = meteor.compute(predictions=pred_str, references=ref_str)
# bleu_res = sacrebleu.compute(predictions=pred_str, references=[[r] for r in ref_str])

# # try:
# #     bleurt_res = bleurt.compute(predictions=pred_str, references=ref_str)
# #     bleurt_mean = float(np.mean(bleurt_res["scores"]))
# # except Exception:
# #     bleurt_mean = float("nan")

# ##############################################################
# # 12. AGGREGATE & SAVE
# ##############################################################
# metrics = {
#     "curve": curve_name,
#     "rouge1": rouge_res["rouge1"],
#     "rouge2": rouge_res["rouge2"],
#     "rougeL": rouge_res["rougeL"],
#     "rougeLsum": rouge_res["rougeLsum"],
#     "bertscore_f1": float(np.mean(bert_res["f1"])),
#     "meteor": float(meteor_res["meteor"]),
#     "bleu": float(bleu_res["score"]),
#     # "bleurt": bleurt_mean,
# }

# if isinstance(unieval_scores, list):
#     for k in unieval_scores[0]:
#         metrics[f"unieval_{k}"] = float(np.mean([d[k] for d in unieval_scores]))

# metrics_path = f"{OUTDIR}/vanilla_llama70b_{curve_name}_metrics.json"
# with open(metrics_path, "w") as f:
#     json.dump(metrics, f, indent=2)

# print("\n✅ DONE")
# print(json.dumps(metrics, indent=2))
# print(f"\n📁 Predictions: {pred_csv}")
# print(f"📁 Metrics:     {metrics_path}")








# -*- coding: utf-8 -*-
"""
Few-shot structured note generation + evaluation
(Vanilla LLaMA / DeepSeek / Qwen)

Uses ONE few-shot CSV per run (learning curve point)
Evaluates with UniEval + ROUGE / BERTScore / METEOR / BLEU / BLEURT
"""

##############################################################
# 1. BLOCK GPU0 (IMPORTANT)
##############################################################
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

##############################################################
# 2. IMPORTS
##############################################################
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import evaluate
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich import print
from rich.progress import track

##############################################################
# 3. GPU SELECTION
##############################################################
available_gpus = [0, 1, 2, 3]
device_id = random.choice(available_gpus)
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(f"✅ Using logical GPU {device_id} (physical GPU {device_id+1})")

##############################################################
# 4. PATHS
##############################################################
OUTPUT_DIR = "/home/raza/EMRV1/EMRV2/Baseline/vanilla/vanilla_llama70b"
UNIEVAL_PATH = "/home/raza/EMRV1/EMRV2/UniEval"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(UNIEVAL_PATH)

##############################################################
# 5. METRICS
##############################################################
from utils import convert_to_json
from metric.evaluator import get_evaluator

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")

# # BLEURT (safe)
# try:
#     bleurt = evaluate.load("bleurt", config_name="bleurt-20")
# except Exception as e:
#     print("⚠️ BLEURT disabled:", e)
#     bleurt = None

##############################################################
# 6. MODEL LOADER
##############################################################
def load_model(model_name):
    model_map = {
        "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "deepseek-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "qwen-7b": "Qwen/Qwen2-7B-Instruct",
    }
    model_id = model_map.get(model_name.lower(), model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

##############################################################
# 7. DATA LOADING
##############################################################
def load_test_dataset(dataset_name):
    if dataset_name == "endocrinology":
        data = load_dataset(
            "starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper"
        )["train"]
        return pd.DataFrame({
            "transcript": data["transcript"],
            "structured_note": data["structured_note"]
        })

    elif dataset_name == "aci-bench":
        df = pd.read_csv(
            "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"
        )
        if "dialogue" in df.columns:
            df = df.rename(columns={"dialogue": "transcript", "note": "structured_note"})
        return df

    else:
        raise ValueError("Unknown dataset")

##############################################################
# 8. FEW-SHOT PROMPT
##############################################################
def build_prompt(fewshot_df, dialogue):
    prompt = "You are a clinical documentation assistant.\n\n"
    for _, row in fewshot_df.iterrows():
        prompt += (
            "TRANSCRIPT:\n" + row["dialogue"].strip() + "\n\n"
            "STRUCTURED NOTE:\n" + row["note"].strip() + "\n\n###\n\n"
        )
    prompt += f"TRANSCRIPT:\n{dialogue.strip()}\n\nSTRUCTURED NOTE:\n"
    return prompt

##############################################################
# 9. GENERATION
##############################################################
def generate_fewshot(model_name, dataset_name, fewshot_csv, temperature):
    tokenizer, model = load_model(model_name)
    test_df = load_test_dataset(dataset_name)
    fewshot_df = pd.read_csv(fewshot_csv)

    predictions = []

    for dlg in track(test_df["transcript"], description="Generating"):
        prompt = build_prompt(fewshot_df, dlg)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1032).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1032,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded.split("STRUCTURED NOTE:")[-1].strip())

    test_df["generated_note"] = predictions

    curve_name = Path(fewshot_csv).stem
    out_csv = Path(OUTPUT_DIR) / f"{dataset_name}_fewshot_{model_name}_{curve_name}.csv"
    test_df.to_csv(out_csv, index=False)
    print(f"📁 Saved predictions → {out_csv}")

    return test_df, out_csv

##############################################################
# 10. EVALUATION
##############################################################
def evaluate_outputs(df, out_csv):
    preds = df["generated_note"].astype(str).tolist()
    refs = df["structured_note"].astype(str).tolist()
    srcs = df["transcript"].astype(str).tolist()

    print("🔹 UniEval")
    data_json = convert_to_json(preds, srcs, refs)
    evaluator = get_evaluator("summarization", device=device)
    unieval_scores = evaluator.evaluate(data_json)

    print("🔹 Classic metrics")
    metrics = {
        "rouge1": rouge.compute(predictions=preds, references=refs)["rouge1"],
        "rouge2": rouge.compute(predictions=preds, references=refs)["rouge2"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "meteor": meteor.compute(predictions=preds, references=refs)["meteor"],
        "bleu": sacrebleu.compute(predictions=preds, references=[[r] for r in refs])["score"],
        "bertscore_f1": float(np.mean(
            bertscore.compute(
                predictions=preds,
                references=refs,
                lang="en",
                model_type="roberta-large",
                device=f"cuda:{device_id}"
            )["f1"]
        ))
    }

    # if bleurt:
    #     try:
    #         metrics["bleurt"] = float(np.mean(
    #             bleurt.compute(predictions=preds, references=refs)["scores"]
    #         ))
    #     except Exception:
    #         metrics["bleurt"] = float("nan")

    for k in unieval_scores[0]:
        metrics[f"unieval_{k}"] = float(np.mean([d[k] for d in unieval_scores]))

    out_json = out_csv.with_suffix(".metrics.json")
    pd.Series(metrics).to_json(out_json, indent=2)

    print("\n✅ Final Metrics")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

    print(f"📁 Saved metrics → {out_json}")

##############################################################
# 11. MAIN
##############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fewshot_csv", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()

    df, out_csv = generate_fewshot(
        args.model,
        args.dataset,
        args.fewshot_csv,
        args.temperature
    )
    evaluate_outputs(df, out_csv)
