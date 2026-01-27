
# python /Baseline/experiments.py --model llama-3-8b --sample_size 15 --temperature 0.2 --k 3
# -*- coding: utf-8 -*-
"""
ICL generation and evaluation for Endocrinology dataset
Supports GPT-4, LLaMA-3, DeepSeek, and Qwen models.
"""

import os
import re
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from rich import print
from rich.progress import track
import evaluate
import sys
from typing import List

# UniEval import
sys.path.append('/UniEval')
from utils import convert_to_json
from metric.evaluator import get_evaluator

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ===============================================================
# CONFIG
# ===============================================================
MAX_INPUT_TOKENS = 6192
MAX_OUTPUT_TOKENS = 600
OUTPUT_DIR = "/Baseline/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================================================
# FETCH TOP-K CONTEXTS
# ===============================================================
def fetch_in_context_examples(train, test, k=3):
    embedder = SentenceTransformer("hkunlp/instructor-large")
    instruction = "Represent the medical dialogues for clustering:"
    print("[green]Encoding embeddings for retrieval...[/green]")

    test_emb = embedder.encode(
        [[instruction, f"dialogue: {t}"] for t in test["dialogue"]],
        show_progress_bar=True,
    )
    train_emb = embedder.encode(
        [[instruction, f"dialogue: {t}"] for t in train["dialogue"]],
        show_progress_bar=True,
    )

    top_k_indices = []
    for i in range(len(test_emb)):
        scores = util.cos_sim(np.expand_dims(test_emb[i], 0), train_emb)
        top_k_idx = torch.topk(scores, k=min(k, len(scores))).indices.flatten().tolist()
        top_k_indices.append(top_k_idx)
    return top_k_indices

# ===============================================================
# FORMAT EXAMPLES
# ===============================================================
def format_in_context_example(train, idx, include_dialogue=False):
    example = ""
    if include_dialogue:
        example += f"\nEXAMPLE DIALOGUE:\n{train['dialogue'][idx].strip()}"
    example += f"\nEXAMPLE STRUCTURED NOTE:\n{train['note'][idx].strip()}"
    return example

# ===============================================================
# LOAD MODEL (GPT or HF)
# ===============================================================
def load_model(model_name: str, temperature=0.7):
    """Load GPT-4 via OpenAI or open-weight model via Transformers."""
    if model_name.lower() == "gpt-4":
        llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)
        tokenizer = None
        model = None
    else:
        model_id_map = {
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
            "deepseek-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "qwen-9b": "Qwen/Qwen2-7B-Instruct",
        }
        model_id = model_id_map.get(model_name.lower(), model_name)
        print(f"[green]Loading model: {model_id}[/green]")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        llm = None
    return llm, tokenizer, model

# ===============================================================
# GENERATE NOTES
# ===============================================================
def generate_predictions(model_name="gpt-4", k=3, temperature=0.7, include_dialogue=True, sample_size=20):
    print(f"[cyan]Loading ACI-Bench dataset locally... using model: {model_name}[/cyan]")

    # Define your local dataset paths
    train_path = "/datasets/ACI-TaskB-TrainingSet.csv"
    test_path  = "/datasets/ACI-TaskB-TestingSet.csv"

    # Load using pandas instead of huggingface hub
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Ensure expected columns exist
    required_cols = {"dialogue", "note"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(test_df.columns):
        raise ValueError(f"❌ Missing required columns in dataset. Expected {required_cols}")

    # Convert to Dataset-like dict for compatibility
    train = train_df.to_dict(orient="list")
    test  = test_df.to_dict(orient="list")

    print(f"[green]Loaded local ACI-Bench dataset: train={len(train_df)} test={len(test_df)}[/green]")


    if sample_size and sample_size < len(test):
        test = test.shuffle(seed=42).select(range(sample_size))
        print(f"[yellow]Subsampled to {len(test)} test samples.[/yellow]")

    llm, tokenizer, model = load_model(model_name, temperature)
    prompt_template = (
        "Write a structured clinical note based on this doctor-patient transcript. "
        "Follow the structure and level of detail from the example notes. "
        "Do not fabricate information.\n"
        "{examples}\n\nTRANSCRIPT: {dialogue}\nSTRUCTURED NOTE:\n"
    )

    top_k_indices = fetch_in_context_examples(train, test, k=k)
    predictions: List[str] = []

    for i, dialogue in track(enumerate(test["dialogue"]), total=len(test["dialogue"]), description=f"Generating with {model_name}"):
        examples = ""
        for idx in top_k_indices[i]:
            examples += format_in_context_example(train, idx, include_dialogue=include_dialogue)
        prompt = prompt_template.format(examples=examples, dialogue=dialogue)

        if llm:  # GPT-4 path
            note = llm.invoke(prompt).content.strip()
        else:    # open-weight LLM path
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_OUTPUT_TOKENS,
                    temperature=temperature,
                )
            # Decode and clean output
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Keep only text after "STRUCTURED NOTE:"
            if "STRUCTURED NOTE:" in full_text:
                note = full_text.split("STRUCTURED NOTE:", 1)[-1].strip()
            else:
                # Fallback if model doesn’t include that token
                note = full_text.replace(prompt, "").strip()

        predictions.append(note)

    # ✅ Only store generated structured note (no examples or prompt)
    df_out = pd.DataFrame({
        "transcript": test["dialogue"],
        "structured_note": test["note"],
        "generated_note": predictions,
    })

    out_csv = Path(OUTPUT_DIR) / f"aci-bench_icl_predictions_{model_name}_temp{temperature}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[green]Predictions saved to {out_csv}[/green]")
    return df_out


# ===============================================================
# EVALUATION
# ===============================================================
def evaluate_predictions(df: pd.DataFrame, model_name: str, temperature: float):
    preds = df["generated_note"].tolist()
    refs = df["structured_note"].tolist()
    dialogues = df["transcript"].tolist()

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")
    sacrebleu = evaluate.load("sacrebleu")
    bleurt = evaluate.load("bleurt", config_name="bleurt-20")

    data = convert_to_json(output_list=preds, src_list=dialogues, ref_list=refs)
    evaluator = get_evaluator("summarization")
    eval_scores = evaluator.evaluate(data, print_result=True)

    rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="en")
    meteor_scores = meteor.compute(predictions=preds, references=refs)
    bleu_scores = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])
    try:
        bleurt_scores = bleurt.compute(predictions=preds, references=refs)
        bleurt_mean = float(np.mean(bleurt_scores["scores"]))
    except Exception:
        bleurt_mean = float("nan")

    metrics = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
        "meteor": meteor_scores["meteor"],
        "bleu": bleu_scores["score"],
        "bertscore_f1": float(np.mean(bert_scores["f1"])),
        "bleurt": bleurt_mean,
    }

    if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
        avg_uni = {f"unieval_{k}": float(np.mean([d[k] for d in eval_scores])) for k in eval_scores[0].keys()}
        metrics.update(avg_uni)
    elif isinstance(eval_scores, dict):
        metrics.update({f"unieval_{k}": v for k, v in eval_scores.items()})

    print(f"\n[bold cyan]Evaluation Results for {model_name} (T={temperature}):[/bold cyan]")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_fp = Path(OUTPUT_DIR) / f"aci-bench_icl_eval_results_{model_name}_temp{temperature}.json"
    pd.Series(metrics).to_json(out_fp, indent=2)
    print(f"[green]Saved metrics to {out_fp}[/green]")
    return metrics

# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    import argparse
    print(">>> Starting ACI-Bench experiment runner...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="Model name: gpt-4, llama-3-8b, deepseek-8b, qwen-9b, etc.")
    parser.add_argument("--sample_size", type=int, default=15,
                        help="Number of test samples to use.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature for generation.")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of in-context examples to use (default: 3).")
    args = parser.parse_args()

    # Default temperature by model family
    model_name = args.model.lower()
    if args.temperature is None:
        if "gpt" in model_name:
            args.temperature = 0.7
        elif "qwen" in model_name:
            args.temperature = 0.3
        else:
            args.temperature = 0.2

    print(f">>> Using model: {args.model}")
    print(f">>> Sample size: {args.sample_size}")
    print(f">>> Temperature: {args.temperature}")
    print(f">>> In-context examples (k): {args.k}")

    try:
        df_out = generate_predictions(
            model_name=args.model,
            k=args.k,
            temperature=args.temperature,
            sample_size=args.sample_size,
        )
        evaluate_predictions(df_out, args.model)
    except Exception as e:
        import traceback
        print(">>> ERROR during experiment:")
        traceback.print_exc()



