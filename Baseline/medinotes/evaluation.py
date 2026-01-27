# # File: evaluation.py (FIXED)
# """
# Evaluation script for MediNotes LoRA + QLoRA models.

# Fixes:
# - bitsandbytes + PEFT inference crash
# - memory_efficient_backward missing
# - LoRA injection failure

# Loads:
# 1) Base model
# 2) Applies LoRA adapter
# 3) Merges LoRA (safe inference)
# """

# import argparse
# import json
# from tqdm import tqdm
# import torch

# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
# )

# from peft import PeftModel
# import os
# # from datasets import load_metric
# import bert_score
# from evaluate import load as load_metric

# # ------------------------------
# # Keep TF quiet & off the GPUs (for BLEURT)
# # ------------------------------
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# try:
#     import tensorflow as tf
#     try: tf.config.set_visible_devices([], 'GPU')
#     except Exception: pass
#     try:
#         tf.config.threading.set_intra_op_parallelism_threads(1)
#         tf.config.threading.set_inter_op_parallelism_threads(1)
#     except Exception: pass
# except Exception:
#     pass

# # from bleurt import score as bleurt_score


# # ---------------------------------------------------------
# # Load model safely: BASE → LoRA → MERGE
# # ---------------------------------------------------------
# # def load_model(model_dir, base_model_name):
# #     print("🔹 Loading tokenizer...")
# #     tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

# #     # Base LLaMA model (FP16)
# #     print(f"🔹 Loading BASE model: {base_model_name}")
# #     base = AutoModelForCausalLM.from_pretrained(
# #         base_model_name,
# #         torch_dtype=torch.float16,
# #         device_map="auto",
# #         load_in_8bit=False,
# #         load_in_4bit=False
# #     )

# #     # LoRA adapter files
# #     adapter_model_file = os.path.join(model_dir, "adapter_model.safetensors")
# #     adapter_config_file = os.path.join(model_dir, "adapter_config.json")

# #     if os.path.exists(adapter_model_file):
# #         print("🔹 LoRA adapter detected. Loading adapter...")
# #         model = PeftModel.from_pretrained(
# #             base,
# #             model_dir,
# #             torch_dtype=torch.float16
# #         )
# #         print("🔹 Merging LoRA → base model...")
# #         model = model.merge_and_unload()
# #     else:
# #         print("⚠️ No adapter found → running base model only")
# #         model = base

# #     model.eval()
# #     return model, tokenizer

# # ----------------------------------------------------------------------------------#
# # Load model safely: BASE → LoRA → MERGE  (For Qwen Model) otherwise commentted out #
# # ----------------------------------------------------------------------------------#
# def load_model(model_dir, base_model_name):
#     print("🔹 Loading tokenizer (from checkpoint)...")
#     tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

#     print(f"🔹 Loading BASE model: {base_model_name}")
#     base = AutoModelForCausalLM.from_pretrained(
#         base_model_name,
#         torch_dtype=torch.float16,
#         device_map="auto",
#     )

#     # Match the vocab size used during training
#     print("🔹 Resizing token embeddings to match training tokenizer...")
#     base.resize_token_embeddings(len(tokenizer))

#     # Load LoRA adapter
#     adapter_model_file = os.path.join(model_dir, "adapter_model.safetensors")
#     if os.path.exists(adapter_model_file):
#         print("🔹 LoRA adapter detected. Loading adapter...")
#         model = PeftModel.from_pretrained(base, model_dir)
#         print("🔹 Merging LoRA → base...")
#         model = model.merge_and_unload()
#     else:
#         print("⚠️ No LoRA adapter found. Using base model.")
#         model = base

#     model.eval()
#     return model, tokenizer





# # ---------------------------------------------------------
# # Generation
# # ---------------------------------------------------------
# def generate_note(model, tokenizer, instruction, dialogue, max_len=1024):
#     prompt = (
#         f"{instruction}\n\n"
#         f"{dialogue}\n\n"
#         f"### Response:\n"
#     )

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=4096
#     ).to(model.device)

#     with torch.no_grad():
#         out = model.generate(
#             **inputs,
#             max_new_tokens=max_len,
#             do_sample=False,
#             temperature=0.0,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     txt = tokenizer.decode(out[0], skip_special_tokens=True)
#     if "### Response:" in txt:
#         txt = txt.split("### Response:")[-1].strip()
#     return txt


# # ---------------------------------------------------------
# # Evaluation
# # ---------------------------------------------------------
# def run_eval(model, tokenizer, test_data, output_file):
#     refs, preds = [], []

#     for ex in tqdm(test_data):
#         pred = generate_note(model, tokenizer, ex["instruction"], ex["input"])
#         preds.append(pred)
#         refs.append(ex["output"])

#     print("🔸 Computing ROUGE...")
#     rouge = load_metric("rouge")
#     rouge_scores = rouge.compute(predictions=preds, references=refs)

#     print("🔸 Computing METEOR...")
#     meteor = load_metric("meteor")
#     meteor_score = meteor.compute(predictions=preds, references=refs)

#     print("🔸 Computing BERTScore...")
#     _, _, F = bert_score.score(preds, refs, lang="en", verbose=False)
#     bert_f1 = F.mean().item()

#     print("🔸 Computing BLEURT...")
#     sacrebleu = load_metric("sacrebleu")
#     bleurt = load_metric("bleurt", config_name="bleurt-20")
#     # bleurt = bleurt_score.BleurtScorer("bleurt-large-512")
#     bleurt_values = bleurt.compute(predictions=preds, references=refs)["scores"]
#     bleurt_mean = sum(bleurt_values) / len(bleurt_values)

#     results = {
#         "rouge1": rouge_scores["rouge1"],
#         "rouge2": rouge_scores["rouge2"],
#         "rougeL": rouge_scores["rougeL"],
#         "meteor": meteor_score["meteor"],
#         "bertscore_f1": bert_f1,
#         "bleurt": bleurt_mean,
#     }

#     with open(output_file, "w") as f:
#         json.dump({
#             "results": results,
#             "predictions": preds,
#             "references": refs
#         }, f, indent=2)

#     print("\n✅ Evaluation Completed!")
#     print(json.dumps(results, indent=2))


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--model_dir",
#         type=str,
#         required=True,
#         help="Directory containing LoRA adapter + tokenizer"
#     )

#     parser.add_argument(
#         "--base_model",
#         type=str,
#         required=True,
#         help="Base pretrained model name (e.g. meta-llama/Meta-Llama-3-8B-Instruct)"   #deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#     )

#     parser.add_argument(
#         "--test_file",
#         type=str,
#         required=True,
#         help="Test CSV file"
#     )

#     parser.add_argument(
#         "--output_file",
#         type=str,
#         required=True,
#         help="Where to save evaluation results JSON"
#     )

#     args = parser.parse_args()

#     model, tokenizer = load_model(args.model_dir, args.base_model)



#     test_data = []
#     with open(args.test_file, "r") as f:
#         for line in f:
#             item = json.loads(line)
#             if item["split"] == "test":
#                 test_data.append(item)

#     run_eval(model, tokenizer, test_data, args.output_file)


# if __name__ == "__main__":
#     main()






"""
Universal Evaluation Script for MediNotes + Endocrinology LoRA/QLoRA Models
Supports:
    - LLaMA family
    - Qwen2
    - DeepSeek
    - Any PEFT LoRA checkpoint
Metrics:
    - ROUGE
    - METEOR
    - BLEURT
    - BERTScore
    - UniEval (summarization)

"""

import argparse
import json
import torch
import os
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel
import bert_score
from evaluate import load as load_metric

# UniEval imports
import sys
sys.path.append("/home/raza/EMRV1/EMRV2/UniEval")
from utils import convert_to_json
from metric.evaluator import get_evaluator


# Disable TF GPU for BLEURT
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
    try: tf.config.set_visible_devices([], 'GPU')
    except: pass
except:
    pass


# =====================================================================
# MODEL LOADING (SAFE FOR QLoRA + MULTI-GPU TRAINING)
# =====================================================================
def load_model(model_dir, base_model):
    print("🔹 Loading tokenizer from fine-tuned directory...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

    print(f"🔹 Loading BASE model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("🔹 Aligning embedding sizes...")
    base.resize_token_embeddings(len(tokenizer))

    adapter_file = os.path.join(model_dir, "adapter_model.safetensors")

    if os.path.exists(adapter_file):
        print("🔹 LoRA adapter detected → loading...")
        model = PeftModel.from_pretrained(base, model_dir)
        print("🔹 Merging LoRA → Base model...")
        model = model.merge_and_unload()
    else:
        print("⚠️ No LoRA adapter found → using base model.")
        model = base

    model.eval()
    return model, tokenizer


# =====================================================================
# GENERATION
# =====================================================================
def generate_note(model, tokenizer, instruction, dialogue, max_len=1024):
    prompt = f"{instruction}\n\n{dialogue}\n\n### Response:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

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


# =====================================================================
# EVALUATION
# =====================================================================
def run_eval(model, tokenizer, test_data, output_file):

    preds, refs, src_list = [], [], []

    for ex in tqdm(test_data, desc="Generating predictions"):
        pred = generate_note(model, tokenizer, ex["instruction"], ex["input"])
        preds.append(pred)
        refs.append(ex["output"])
        src_list.append(ex["input"])


        clean_preds, clean_refs, clean_src = [], [], []

    for p, r, s in zip(preds, refs, src_list):
        if p is None or len(str(p).strip()) == 0:
            continue
        if r is None or len(str(r).strip()) == 0:
            continue
        if s is None or len(str(s).strip()) == 0:
            continue

        clean_preds.append(p)
        clean_refs.append(r)
        clean_src.append(s)

    preds, refs, src_list = clean_preds, clean_refs, clean_src

    if len(preds) == 0:
        raise RuntimeError("❌ All predictions were empty or invalid. Cannot evaluate.")

    metrics = {}

    # -----------------------------
    # 1. ROUGE
    # -----------------------------
    print("🔸 Computing ROUGE...")
    rouge = load_metric("rouge")
    r = rouge.compute(predictions=preds, references=refs)
    metrics["rouge1"] = r["rouge1"]
    metrics["rouge2"] = r["rouge2"]
    metrics["rougeL"] = r["rougeL"]

    # -----------------------------
    # 2. METEOR
    # -----------------------------
    print("🔸 Computing METEOR...")
    meteor = load_metric("meteor")
    m = meteor.compute(predictions=preds, references=refs)
    metrics["meteor"] = m["meteor"]

    # -----------------------------
    # 3. BERTScore
    # -----------------------------
    print("🔸 Computing BERTScore...")
    _, _, F = bert_score.score(preds, refs, lang="en", verbose=False)
    metrics["bertscore_f1"] = F.mean().item()

    # -----------------------------
    # 4. BLEURT
    # -----------------------------
    print("🔸 Computing BLEURT...")
    bleurt = load_metric("bleurt", config_name="bleurt-20")
    b = bleurt.compute(predictions=preds, references=refs)
    metrics["bleurt"] = sum(b["scores"]) / len(b["scores"])

    # -----------------------------
    # 5. UniEval
    # -----------------------------
    print("🔹 Running UniEval...")
    data_json = convert_to_json(
        output_list=preds,
        src_list=src_list,
        ref_list=refs
    )

    evaluator = get_evaluator("summarization", device="cuda")
    uni = evaluator.evaluate(data_json, print_result=False)

    # UniEval output aggregation
    for key in uni[0].keys():
        metrics[f"unieval_{key}"] = float(sum(d[key] for d in uni) / len(uni))

    # -----------------------------
    # SAVE
    # -----------------------------
    with open(output_file, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "predictions": preds,
                "references": refs,
            },
            f,
            indent=2
        )

    print("\n✅ Evaluation complete!")
    print(json.dumps(metrics, indent=2))


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.base_model)

    # Load test instances
    test_data = []
    with open(args.test_file, "r") as f:
        for line in f:
            j = json.loads(line)
            if j["split"] == "test":
                test_data.append(j)

    run_eval(model, tokenizer, test_data, args.output_file)


if __name__ == "__main__":
    main()
