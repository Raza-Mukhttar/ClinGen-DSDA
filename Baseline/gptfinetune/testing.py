# Usage: python /Baseline/gptfinetune/testing.py 
# --model_dir /Baseline/gptfinetune/checkpoints/llama_70b 
# --base_model meta-llama/Meta-Llama-3-70B-Instruct or deepseek-ai/DeepSeek-R1-Distill-Llama-70B or Qwen/Qwen2-7B-Instruct
# --test_file /Baseline/gptfinetune/data/valid.jsonl 
# --output_file /Baseline/gptfinetune/outputs/eval_aci_llama_70B.json


#!/usr/bin/env python3
import os
import json
import argparse
from dataclasses import dataclass
from tqdm import tqdm

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
# import evaluate
import bert_score
from evaluate import load as load_metric

# UniEval imports
import sys
sys.path.append("/UniEval")
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

##############################################################################
# ---------------------------- DATASET LOADER ------------------------------ #
##############################################################################

ASSISTANT_ALIASES = {
    "assistant", "bot", "ai", "gpt",
    "assistant_response", "assistant_reply",
    "response", "reply"
}

def is_assistant(role: str):
    if not isinstance(role, str):
        return False
    r = role.strip().lower()
    return r in ASSISTANT_ALIASES or r.endswith("assistant")


def messages_to_text(messages, tokenizer=None):
    try:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except:
        pass

    out = []
    for m in messages:
        role = m.get("role", "").strip()
        content = m.get("content", "").strip()
        out.append(f"[{role}]: {content}")
    return "\n".join(out)


def load_eval_dataset(path, tokenizer=None):
    """
    Returns a list of dicts:
    {
        "prompt": <text>,
        "reference": <assistant response>,
        "source": <original dialogue>
    }
    """

    data = []
    skipped = 0
    total = 0

    with open(path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                j = json.loads(line)
            except Exception as e:
                skipped += 1
                print(f"[WARN] invalid JSON line {i}: {e}")
                continue

            messages = j.get("messages")
            if not isinstance(messages, list):
                skipped += 1
                print(f"[WARN] no messages at line {i}")
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
                print(f"[WARN] no assistant on line {i}")
                continue

            # build prompt = everything BEFORE assistant
            prompt_messages = messages[:last_ass_idx]
            reference_message = messages[last_ass_idx]

            completion = reference_message.get("content", "").strip()
            if completion == "":
                skipped += 1
                print(f"[WARN] empty assistant content at {i}")
                continue

            prompt_text = messages_to_text(prompt_messages, tokenizer)

            data.append({
                "prompt": prompt_text,
                "reference": completion,
                "source": j
            })

    print(f"[INFO] Evaluation dataset loaded: total={total}, kept={len(data)}, skipped={skipped}")

    if len(data) == 0:
        raise RuntimeError("Dataset loader produced 0 usable samples for evaluation.")

    return data


##############################################################################
# ---------------------------- MODEL LOADER ------------------------------- #
##############################################################################

def load_model(model_dir, base_model):
    print("🔹 Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    print("🔹 Enabling 4-bit weights (same as training)…")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("🔹 Loading base model shards…")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("🔹 Loading LoRA adapter…")
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    return model, tokenizer


##############################################################################
# ---------------------------- EVALUATION ---------------------------------- #
##############################################################################

def generate_prediction(model, tokenizer, prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    pred = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return pred.strip()


def run_eval(model, tokenizer, data, output_path):
    preds = []
    refs = []
    srcs = []

    print("🔹 Generating predictions…")
    for ex in tqdm(data):
        pred = generate_prediction(model, tokenizer, ex["prompt"])
        preds.append(pred)
        refs.append(ex["reference"])
        srcs.append(ex["prompt"])


    clean_preds, clean_refs, clean_src = [], [], []

    for p, r, s in zip(preds, refs, srcs):
        if p is None or len(str(p).strip()) == 0:
            continue
        if r is None or len(str(r).strip()) == 0:
            continue
        if s is None or len(str(s).strip()) == 0:
            continue

        clean_preds.append(p)
        clean_refs.append(r)
        clean_src.append(s)

    preds, refs, srcs = clean_preds, clean_refs, clean_src

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
        src_list=srcs,
        ref_list=refs
    )

    evaluator = get_evaluator("summarization", device="cuda")
    uni = evaluator.evaluate(data_json, print_result=False)

    # UniEval output aggregation
    for key in uni[0].keys():
        metrics[f"unieval_{key}"] = float(sum(d[key] for d in uni) / len(uni))


    result = {
        "scores": metrics,
        "predictions": preds,
        "references": refs,
        
    }

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Evaluation complete. Saved to: {output_path}")


##############################################################################
# -------------------------------- MAIN ------------------------------------ #
##############################################################################

def parse_args():
    A = argparse.ArgumentParser()
    A.add_argument("--model_dir", type=str, required=True)
    A.add_argument("--base_model", type=str, required=True)
    A.add_argument("--test_file", type=str, required=True)
    A.add_argument("--output_file", type=str, required=True)
    return A.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model, tokenizer = load_model(args.model_dir, args.base_model)

    test_data = load_eval_dataset(args.test_file, tokenizer)

    run_eval(model, tokenizer, test_data, args.output_file)
