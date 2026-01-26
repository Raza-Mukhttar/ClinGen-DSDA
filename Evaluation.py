import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
import os
import re
import sys




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Stage2_BiasedAttentionEncoder import BiasedAttentionEncoder
from Stage3_PromptInjection import PromptProjection
from DataPreprocessing import preprocess_pipeline
from Stage4_LammaTraining import PromptedLlamaMultiTask

sys.path.append('/UniEval')
from utils import convert_to_json
from metric.evaluator import get_evaluator

import evaluate
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
evaluator = get_evaluator('summarization')

SECTION_HEADERS = [
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "REVIEW OF SYSTEMS",
    "PHYSICAL EXAMINATION",
    "RESULTS",
    "ASSESSMENT",
    "PLAN",
    "ASSESSMENT AND PLAN",
    "INSTRUCTIONS"
]

def extract_sections(text, section_headers=SECTION_HEADERS):
    sections = {}
    pattern = '|'.join([re.escape(h) for h in section_headers])
    parts = re.split(f"({pattern})", text)
    curr_section = None
    for part in parts:
        part = part.strip()
        if part in section_headers:
            curr_section = part
            sections[curr_section] = ""
        elif curr_section and part:
            if sections[curr_section]:
                sections[curr_section] += "\n" + part
            else:
                sections[curr_section] = part
    for h in section_headers:
        if h not in sections:
            sections[h] = ""
    return sections

def canonicalize_note(text, section_headers=SECTION_HEADERS):
    sections = extract_sections(text, section_headers)
    canonical = ""
    for h in section_headers:
        canonical += h + "\n\n" + sections[h].strip() + "\n\n"
    return canonical.strip()

def normalize_section_headers(note_body):
    for header in SECTION_HEADERS:
        pattern = rf"(?im)^{header.replace(' ', '[ _]')}[ \t]*:?"
        note_body = re.sub(pattern, header, note_body)
    return note_body

def postprocess_generated_note(generated_text):
    note_match = re.search(r"\[NOTE\]", generated_text, re.IGNORECASE)
    if not note_match:
        return generated_text.strip()
    note_start = note_match.end()
    after_note = generated_text[note_start:]
    end_match = re.search(r"\[END OF NOTE\]|\[END NOTE\]|\[END INSTRUCTION\]|\[INSTRUCTION\]", after_note, re.IGNORECASE)
    if end_match:
        note_body = after_note[:end_match.start()]
    else:
        note_body = after_note
    note_body = normalize_section_headers(note_body)
    note_body = re.sub(r'\n{2,}', '\n\n', note_body)
    return note_body.strip("\n\r\t ")

# --------- CONFIG ---------
llama_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
test_csv = "/datasets/ACI-TaskB-TestingSet.csv"
model_safetensors_path = "/outputs/model_training_outputs/llama_joint_summ_ckpt/final/model.safetensors"
stage1_model_dir = "/outputs/stage1_outputs/final_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
output_csv = "/outputs/test_outputs/test_results.csv"

tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(llama_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
hidden_size = base_model.config.hidden_size
prompt_len = 16
soft_prompt_len = 16
ner_label_count = 7

stage1_model = AutoModelForTokenClassification.from_pretrained(stage1_model_dir).to(device)
stage1_tokenizer = AutoTokenizer.from_pretrained(stage1_model_dir)
backbone_name = "emilyalsentzer/Bio_ClinicalBERT"
hidden_size = stage1_model.config.hidden_size
entity_label_count = stage1_model.config.num_labels
stage2_encoder = BiasedAttentionEncoder(
    backbone_name=backbone_name,
    hidden_size=hidden_size,
    entity_label_count=entity_label_count,
    bias_strength=3.0
).to(device)
stage3_proj = PromptProjection(
    hidden_size=hidden_size,
    prompt_len=16,
    mode="attentive"
)

# --------- PREPROCESS TEST DATA (WITH CACHING) ---------
test_df = pd.read_csv(test_csv)
test_dialogues = test_df["dialogue"].astype(str).tolist()
test_notes = test_df.get("note", pd.Series([""] * len(test_df))).astype(str).tolist() 

test_pt_path = "/outputs/preprocessing_outputs/test_data_list.pt"
if os.path.exists(test_pt_path):
    print("Loading preprocessed test data...")
    test_data_list = torch.load(test_pt_path)
else:
    print("Preprocessing test set...")
    test_data_list = preprocess_pipeline(
        test_dialogues, test_notes,
        stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
        tokenizer, device,
    )
    torch.save(test_data_list, test_pt_path)
    print(f"Saved preprocessed test data to {test_pt_path}")

# --------- LOAD TRAINED MODEL FROM SAFETENSORS ---------

model = PromptedLlamaMultiTask(
    base_model=base_model,
    prompt_len=prompt_len,
    hidden_size=hidden_size,
    ner_label_count=ner_label_count,
    soft_prompt_len=soft_prompt_len
)
state_dict = load_safetensors(model_safetensors_path)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

results_data = []
all_metrics = {
    "rouge1": [],
    "rouge2": [],
    "rougeL": [],
    "rougeLsum": [],
    "bertscore_f1": [],
    
}



unieval_metric_names = None
unieval_metrics_all = {}

for idx, sample in enumerate(tqdm(test_data_list, desc="Inference")):
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
    prompt_embeds = sample['prompt_embeds'].unsqueeze(0).to(device)

    # --------- GENERATION (PROMPT-ONLY GENERATION) ---------
    with torch.no_grad():
        # The new prediction_step uses only the prompt!
        labels = sample.get('labels', None)
        if labels is not None:
            labels = labels.unsqueeze(0)
            labels_mask = (labels[0] == -100)
            prompt_token_count = torch.sum(labels_mask).item()
        else:
            # fallback: use prompt_len if labels missing
            prompt_token_count = prompt_len

        prompt_input_ids = input_ids[:, :prompt_token_count]
        prompt_attention_mask = attention_mask[:, :prompt_token_count]

        gen_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": 512,
            "early_stopping": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "num_beams": 2,
            "do_sample": False
        }
        generated_tokens = model.llama.generate(**gen_kwargs)
        pred_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        pred_str = postprocess_generated_note(pred_str)

    ref_str = canonicalize_note(test_notes[idx].strip())

    # --------- METRICS ---------
    rouge_result = rouge.compute(predictions=[pred_str], references=[ref_str], use_stemmer=True)
    bertscore_result = bertscore.compute(predictions=[pred_str], references=[ref_str], lang="en")
    bertscore_f1 = float(bertscore_result["f1"][0])


    

    # UniEval (per sample)
    sample_json = convert_to_json(
        output_list=[pred_str],
        src_list=[test_dialogues[idx]],
        ref_list=[ref_str]
    )
    unieval_score = evaluator.evaluate(sample_json, print_result=False)
    if isinstance(unieval_score, list) and len(unieval_score) > 0 and isinstance(unieval_score[0], dict):
        unieval_score = unieval_score[0]
    if unieval_metric_names is None:
        unieval_metric_names = list(unieval_score.keys())
        for name in unieval_metric_names:
            unieval_metrics_all[name] = []
    for name in unieval_metric_names:
        unieval_metrics_all[name].append(unieval_score[name])

    # Store all metrics for final average
    all_metrics["rouge1"].append(rouge_result["rouge1"])
    all_metrics["rouge2"].append(rouge_result["rouge2"])
    all_metrics["rougeL"].append(rouge_result["rougeL"])
    all_metrics["rougeLsum"].append(rouge_result["rougeLsum"])
    all_metrics["bertscore_f1"].append(bertscore_f1)
    

    # Save per-sample row
    row = {
        "reference_note": ref_str,
        "generated_note": pred_str,
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "rougeLsum": rouge_result["rougeLsum"],
        "bertscore_f1": bertscore_f1,
        
    }
    for name in unieval_metric_names:
        row[f"unieval_{name}"] = unieval_score[name]
    results_data.append(row)

# --------- AGGREGATE & PRINT AVERAGES ---------
print("\n===== Averaged Metrics Across All Test Samples =====")
for key, values in all_metrics.items():
    print(f"{key}: {np.mean(values):.4f}")
for name in unieval_metric_names:
    print(f"unieval_{name}: {np.mean(unieval_metrics_all[name]):.4f}")

# --------- SAVE TO CSV ---------
df_out = pd.DataFrame(results_data)
df_out.to_csv(output_csv, index=False)
print(f"\nSaved all results to {output_csv}")