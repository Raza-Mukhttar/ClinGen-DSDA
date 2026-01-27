# ## Baseline implementation using LangChain and GPT-4 for ACI-Benchmark for clinical note generation



# import os
# import re
# import random
# import numpy as np
# import pandas as pd
# import torch
# from pathlib import Path
# from typing import Any, Dict, List
# from datasets import load_dataset
# from InstructorEmbedding import INSTRUCTOR
# from sentence_transformers import util, SentenceTransformer
# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from rich import print
# from rich.progress import track
# import evaluate
# import sys

# # ---- Add UniEval path ----
# sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')
# from utils import convert_to_json
# from metric.evaluator import get_evaluator


# # ===============================================================
# # CONFIG
# # ===============================================================
# MAX_INPUT_TOKENS = 6192
# MAX_OUTPUT_TOKENS = 2000
# TEAM_NAME = "wanglab"

# # ===============================================================
# # STEP 1: RETRIEVE TOP-K IN-CONTEXT EXAMPLES
# # ===============================================================
# def fetch_in_context_examples(train, test, k=3, retrieve_similar=True, filter_by_dataset=False):
#     if retrieve_similar:
#         embedder = SentenceTransformer("hkunlp/instructor-large")
#         instruction = "Represent the Medicine dialogue for clustering:"
#         test_embeddings = embedder.encode(
#             [[instruction, f"dataset: {d} dialogue: {t}"] for d, t in zip(test["dataset"], test["dialogue"])],
#             show_progress_bar=True,
#         )
#         train_embeddings = embedder.encode(
#             [[instruction, f"dataset: {d} dialogue: {t}"] for d, t in zip(train["dataset"], train["dialogue"])],
#             show_progress_bar=True,
#         )

#     top_k_indices = []
#     for i, dataset_name in enumerate(test["dataset"]):
#         if filter_by_dataset:
#             train_indices = [j for j, ds in enumerate(train["dataset"]) if ds == dataset_name]
#         else:
#             train_indices = list(range(len(train["dataset"])))

#         if retrieve_similar:
#             scores = util.cos_sim(np.expand_dims(test_embeddings[i], 0), train_embeddings[train_indices])
#             top_k_idx = torch.topk(scores, k=min(k, len(scores))).indices.flatten().tolist()
#             top_k_indices.append([train_indices[idx] for idx in top_k_idx])
#         else:
#             top_k_indices.append(random.sample(train_indices, min(k, len(train_indices))))
#     return top_k_indices


# # ===============================================================
# # STEP 2: FORMAT EXAMPLES FOR PROMPT
# # ===============================================================
# def format_in_context_example(train, idx, include_dialogue=False):
#     example = ""
#     if include_dialogue:
#         example += f"\nEXAMPLE DIALOGUE:\n{train['dialogue'][idx].strip()}"
#     example += f"\nEXAMPLE NOTE:\n{train['note'][idx].strip()}"
#     return example


# # ===============================================================
# # STEP 3: GENERATE NOTES (GPT-4)
# # ===============================================================
# def generate_predictions(train_fp, test_fp, output_dir, k=3, temperature=0.2, include_dialogue=False):
#     # Load datasets
#     train = load_dataset("csv", data_files={"train": train_fp})["train"]
#     test = load_dataset("csv", data_files={"test": test_fp})["test"]

#     print(f"[green]Loaded train={len(train)} test={len(test)}[/green]")

#     # LLM setup
#     llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)
#     prompt = PromptTemplate(
#         input_variables=["examples", "dialogue"],
#         template=(
#             "Write a clinical note reflecting this doctor-patient dialogue. "
#             "Use the example notes below to decide the structure of the clinical note. "
#             "Do not make up information.\n"
#             "{examples}\n\nDIALOGUE: {dialogue}\nCLINICAL NOTE:\n"
#         ),
#     )
#     chain = LLMChain(llm=llm, prompt=prompt)

#     # Retrieve top-k similar examples
#     top_k_indices = fetch_in_context_examples(train, test, k=k, retrieve_similar=True, filter_by_dataset=True)

#     predictions = []
#     for i, dialogue in track(enumerate(test["dialogue"]), total=len(test["dialogue"]), description="Generating notes"):
#         examples = ""
#         prompt_length = llm.get_num_tokens(prompt.format(dialogue=dialogue, examples=""))
#         for idx in top_k_indices[i]:
#             example = format_in_context_example(train, idx, include_dialogue=include_dialogue)
#             if (prompt_length + llm.get_num_tokens(example)) < MAX_INPUT_TOKENS:
#                 examples += example
#                 prompt_length += llm.get_num_tokens(example)
#         note = chain.run(dialogue=dialogue, examples=examples)
#         note = re.sub(r"(\n|^)(?!.*\\nRESULTS.*)[Ii][Mm][Aa][Gg][Ii][Nn][Gg](:)?", r"\1RESULTS\nImaging\2", note)
#         predictions.append(note)

#     df_out = pd.DataFrame({
#         "encounter_id": test["encounter_id"],
#         "reference_note": test["note"],
#         "prediction": predictions
#     })
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     output_fp = Path(output_dir) / "icl_predictions.csv"
#     df_out.to_csv(output_fp, index=False)
#     print(f"[green]Predictions saved to {output_fp}[/green]")
#     return df_out


# # ===============================================================
# # STEP 4: EVALUATION METRICS
# # ===============================================================
# def evaluate_predictions(df: pd.DataFrame):
#     preds = df["prediction"].tolist()
#     refs = df["reference_note"].tolist()

#     rouge = evaluate.load("rouge")
#     bertscore = evaluate.load("bertscore")
#     meteor = evaluate.load("meteor")
#     sacrebleu = evaluate.load("sacrebleu")
#     bleurt = evaluate.load("bleurt", config_name="bleurt-20")

#     # UniEval setup
#     val_dialogues = df.get("dialogue", [""] * len(df))
#     data = convert_to_json(output_list=preds, src_list=val_dialogues, ref_list=refs)
#     evaluator = get_evaluator("summarization")
#     eval_scores = evaluator.evaluate(data, print_result=True)

#     # ---- Compute standard metrics ----
#     rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
#     bert_scores = bertscore.compute(predictions=preds, references=refs, lang="en")
#     meteor_scores = meteor.compute(predictions=preds, references=refs)
#     bleu_scores = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])

#     try:
#         bleurt_scores = bleurt.compute(predictions=preds, references=refs)
#         bleurt_mean = float(np.mean(bleurt_scores["scores"]))
#     except Exception:
#         bleurt_mean = float("nan")

#     metrics = {
#         "rouge1": rouge_scores["rouge1"],
#         "rouge2": rouge_scores["rouge2"],
#         "rougeL": rouge_scores["rougeL"],
#         "rougeLsum": rouge_scores["rougeLsum"],
#         "meteor": meteor_scores["meteor"],
#         "bleu": bleu_scores["score"],
#         "bertscore_f1": float(np.mean(bert_scores["f1"])),
#         "bleurt": bleurt_mean,
#     }

#     # ---- UniEval average ----
#     if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
#         avg_uni = {f"unieval_{k}": float(np.mean([d[k] for d in eval_scores])) for k in eval_scores[0].keys()}
#         metrics.update(avg_uni)
#     elif isinstance(eval_scores, dict):
#         metrics.update({f"unieval_{k}": v for k, v in eval_scores.items()})

#     print("[bold cyan]Evaluation Results:[/bold cyan]")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")

#     return metrics


# # ===============================================================
# # MAIN ENTRY POINT
# # ===============================================================
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="In-context learning generation + evaluation")
#     parser.add_argument("--train_fp", required=True)
#     parser.add_argument("--test_fp", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--k", type=int, default=3)
#     parser.add_argument("--include_dialogue", action="store_true")
#     args = parser.parse_args()

#     df_out = generate_predictions(
#         train_fp=args.train_fp,
#         test_fp=args.test_fp,
#         output_dir=args.output_dir,
#         k=args.k,
#         include_dialogue=args.include_dialogue,
#     )

#     results = evaluate_predictions(df_out)






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICL generation and evaluation for Endocrinology dataset (GPT-4).
Uses same pipeline as ACI-Bench baseline but loads
datasets from Hugging Face:
  - train: starfishdata/endocrinology_structured_notes_1500_whisper
  - test:  starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper
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
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from rich import print
from rich.progress import track
import evaluate
import sys

# UniEval import
sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')
from utils import convert_to_json
from metric.evaluator import get_evaluator


# ===============================================================
# CONFIG
# ===============================================================
MAX_INPUT_TOKENS = 6192
MAX_OUTPUT_TOKENS = 600
TEAM_NAME = "wanglab"
OUTPUT_DIR = "/home/raza/EMRV1/EMRV2/Baseline/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================================================
# STEP 1: RETRIEVE TOP-K IN-CONTEXT EXAMPLES
# ===============================================================
def fetch_in_context_examples(train, test, k=3, retrieve_similar=True):
    embedder = SentenceTransformer("hkunlp/instructor-large")
    instruction = "Represent the Endocrinology dialogue for clustering:"

    print("[green]Encoding training and testing dialogues for similarity retrieval...[/green]")
    test_embeddings = embedder.encode(
        [[instruction, f"dialogue: {t}"] for t in test["transcript"]],
        show_progress_bar=True,
    )
    train_embeddings = embedder.encode(
        [[instruction, f"dialogue: {t}"] for t in train["transcript"]],
        show_progress_bar=True,
    )

    top_k_indices = []
    for i in range(len(test_embeddings)):
        scores = util.cos_sim(
            np.expand_dims(test_embeddings[i], 0),
            train_embeddings
        )
        top_k_idx = torch.topk(scores, k=min(k, len(scores))).indices.flatten().tolist()
        top_k_indices.append(top_k_idx)
    return top_k_indices


# ===============================================================
# STEP 2: FORMAT EXAMPLES FOR PROMPT
# ===============================================================
def format_in_context_example(train, idx, include_dialogue=False):
    example = ""
    if include_dialogue:
        example += f"\nEXAMPLE DIALOGUE:\n{train['transcript'][idx].strip()}"
    example += f"\nEXAMPLE STRUCTURED NOTE:\n{train['structured_note'][idx].strip()}"
    return example


# ===============================================================
# STEP 3: GENERATE NOTES (GPT-4)
# ===============================================================
def generate_predictions(k=3, temperature=0.7, include_dialogue=False, sample_size=None):
    print("[cyan]Loading Endocrinology datasets from Hugging Face...[/cyan]")
    train = load_dataset("starfishdata/endocrinology_structured_notes_1500_whisper")["train"]
    test = load_dataset("starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper")["train"]
    print(f"[green]Loaded train={len(train)} test={len(test)}[/green]")


    if sample_size and sample_size < len(test):
        test = test.shuffle(seed=42).select(range(sample_size))
        print(f"[yellow]Subsampled test set to {len(test)} samples for quick evaluation.[/yellow]")


    # GPT-4 setup
    llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS)
    prompt = PromptTemplate(
        input_variables=["examples", "dialogue"],
        template=(
            "Write a structured endocrinology clinical note based on the following doctor-patient transcript. "
            "Follow the format and detail style of the provided examples. "
            "Do not fabricate information.\n"
            "{examples}\n\nTRANSCRIPT: {dialogue}\nSTRUCTURED NOTE:\n"
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Retrieve top-k examples
    top_k_indices = fetch_in_context_examples(train, test, k=k)

    predictions = []
    for i, dialogue in track(enumerate(test["transcript"]), total=len(test["transcript"]), description="Generating notes"):
        examples = ""
        prompt_length = llm.get_num_tokens(prompt.format(dialogue=dialogue, examples=""))
        for idx in top_k_indices[i]:
            example = format_in_context_example(train, idx, include_dialogue=include_dialogue)
            if (prompt_length + llm.get_num_tokens(example)) < MAX_INPUT_TOKENS:
                examples += example
                prompt_length += llm.get_num_tokens(example)

        note = chain.run(dialogue=dialogue, examples=examples)
        note = re.sub(r"(\n|^)(?!.*\\nRESULTS.*)[Ii][Mm][Aa][Gg][Ii][Nn][Gg](:)?", r"\1RESULTS\nImaging\2", note)
        predictions.append(note)

    df_out = pd.DataFrame({
        # "encounter_id": test["encounter_id"] if "encounter_id" in test.column_names else np.arange(len(test)),
        "transcript": test["transcript"],
        "structured_note": test["structured_note"],
        "generated_note": predictions,
    })

    output_fp = Path(OUTPUT_DIR) / "endocrinology_icl_predictions.csv"
    df_out.to_csv(output_fp, index=False)
    print(f"[green]Predictions saved to {output_fp}[/green]")
    return df_out


# ===============================================================
# STEP 4: EVALUATION
# ===============================================================
def evaluate_predictions(df: pd.DataFrame):
    preds = df["generated_note"].tolist()
    refs = df["structured_note"].tolist()
    dialogues = df["transcript"].tolist()

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")
    sacrebleu = evaluate.load("sacrebleu")
    bleurt = evaluate.load("bleurt", config_name="bleurt-20")

    # UniEval
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

    print("\n[bold cyan]Evaluation Results:[/bold cyan]")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_fp = Path(OUTPUT_DIR) / "endocrinology_icl_eval_results.json"
    pd.Series(metrics).to_json(out_fp, indent=2)
    print(f"[green]Saved metrics to {out_fp}[/green]")
    return metrics


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    df_out = generate_predictions(k=3, include_dialogue=True, sample_size=15)
    evaluate_predictions(df_out)
