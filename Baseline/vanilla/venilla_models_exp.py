
# python /Baseline/venilla_models_exp.py   --model llama-3-70b   --dataset aci-bench  --sample_size 15   --temperature 0.7
# -*- coding: utf-8 -*-
"""
Zero-shot structured note generation (no ICL) for ACI-Bench and Endocrinology datasets.
Supports LLaMA, DeepSeek, and Qwen models.
"""

import os
import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich import print
from rich.progress import track
import argparse

# ===============================================================
# CONFIG
# ===============================================================
MAX_OUTPUT_TOKENS = 2000
OUTPUT_DIR = "/Baseline/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================================================
# MODEL LOADER
# ===============================================================
def load_model(model_name: str, temperature: float = 0.7):
    """Load HF models for zero-shot structured note generation."""
    model_id_map = {
        "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "deepseek-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "qwen-7b": "Qwen/Qwen2-7B-Instruct",
    }

    model_id = model_id_map.get(model_name.lower(), model_name)
    print(f"[green]Loading model: {model_id}[/green]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

# ===============================================================
# DATA LOADING
# ===============================================================
def load_dataset_local_or_hf(dataset_name: str, sample_size: int = 20):
    """Load dataset (Endocrinology from HF, ACI-Bench from local CSV)."""
    if dataset_name.lower() == "endocrinology":
        print("[cyan]Loading Endocrinology dataset from Hugging Face...[/cyan]")
        data = load_dataset("starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper")["train"]
        df = pd.DataFrame({
            "transcript": data["transcript"],
            "structured_note": data["structured_note"]
        })

    elif dataset_name.lower() == "aci-bench":
        print("[cyan]Loading ACI-Bench dataset from local CSV...[/cyan]")
        test_path = "/datasets/ACI-TaskB-TestingSet.csv"
        df = pd.read_csv(test_path)

        # Rename columns if needed
        if "dialogue" in df.columns and "note" in df.columns:
            df = df.rename(columns={"dialogue": "transcript", "note": "structured_note"})
        elif "transcript" in df.columns and "structured_note" in df.columns:
            pass
        else:
            raise ValueError("❌ ACI-Bench CSV must contain either (dialogue, note) or (transcript, structured_note) columns.")

    else:
        raise ValueError("❌ Unknown dataset. Use 'endocrinology' or 'aci-bench'.")

    # Subsample
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[yellow]Subsampled to {len(df)} test samples.[/yellow]")

    print(f"[green]Dataset loaded successfully with {len(df)} rows.[/green]")
    return df

# ===============================================================
# ZERO-SHOT GENERATION
# ===============================================================
def generate_notes(model_name: str, dataset_name: str, temperature: float, sample_size: int):
    df = load_dataset_local_or_hf(dataset_name, sample_size)
    tokenizer, model = load_model(model_name, temperature)

    predictions = []
    prompt_template = (
        
        "write a clinical note.\n\n"
        "TRANSCRIPT:\n{dialogue}\n\nSTRUCTURED NOTE:\n"
    )

    for dialogue in track(df["transcript"], description=f"Generating ({model_name})", total=len(df)):
        prompt = prompt_template.format(dialogue=dialogue.strip())
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only generated note after the prompt
        generated_note = generated_text.split("STRUCTURED NOTE:")[-1].strip()
        predictions.append(generated_note)

    df["generated_note"] = predictions

    out_path = Path(OUTPUT_DIR) / f"{dataset_name}_zeroshot_predictions_{model_name}_temp{temperature}.csv"
    df.to_csv(out_path, index=False)
    print(f"[green]Saved predictions to {out_path}[/green]")

    return df

# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot LLM evaluation for clinical note generation")
    parser.add_argument("--model", type=str, required=True, help="Model name (llama-3-8b, deepseek-8b, qwen-7b, etc.)")
    parser.add_argument("--dataset", type=str, required=True, choices=["aci-bench", "endocrinology"], help="Dataset to evaluate on")
    parser.add_argument("--sample_size", type=int, default=20, help="Number of test samples")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()
    print(f">>> Running {args.model} on {args.dataset} with temperature={args.temperature}, samples={args.sample_size}")

    try:
        generate_notes(args.model, args.dataset, args.temperature, args.sample_size)
    except Exception as e:
        import traceback
        print(">>> ERROR during generation:")
        traceback.print_exc()
