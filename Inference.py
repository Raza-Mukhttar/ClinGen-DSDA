import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
from safetensors.torch import load_file as load_safetensors
import sys
import os
import re
import evaluate

# --- Import modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Stage2_BiasedAttentionEncoder import BiasedAttentionEncoder
from Stage3_PromptInjection import PromptProjection
from DataPreprocessing import preprocess_pipeline
from Stage4_LammaTraining import PromptedLlamaMultiTask


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

# --- Config ---
llama_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model_safetensors_path = "/outputs/model_training_outputs/llama_joint_summ_ckpt/final/model.safetensors"
stage1_model_dir = "/outputs/stage1_outputs/final_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(llama_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
hidden_size = base_model.config.hidden_size
prompt_len = 16
soft_prompt_len = 16
ner_label_count = 7

stage1_model = AutoModelForTokenClassification.from_pretrained(stage1_model_dir).to("cpu")
stage1_tokenizer = AutoTokenizer.from_pretrained(stage1_model_dir)
backbone_name = "emilyalsentzer/Bio_ClinicalBERT"
hidden_size = stage1_model.config.hidden_size
entity_label_count = stage1_model.config.num_labels
stage2_encoder = BiasedAttentionEncoder(
    backbone_name=backbone_name,
    hidden_size=hidden_size,
    entity_label_count=entity_label_count,
    bias_strength=3.0
).to("cpu")
stage3_proj = PromptProjection(
    hidden_size=hidden_size,
    prompt_len=16,
    mode="attentive"
)

# --- Load Model ---
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

# bleurt = evaluate.load("bleurt")

# --- Inference loop ---
print("Enter dialogue text (multi-line, end with an empty line):")
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)
user_dialogue = "\n".join(lines)
if not user_dialogue.strip():
    print("No input provided. Exiting.")
    exit(0)

# --- Prepare single-sample data ---
test_dialogues = [user_dialogue]
test_notes = [""]  # dummy

print("Processing input...")
test_data_list = preprocess_pipeline(
    test_dialogues, test_notes,
    stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
    tokenizer, device,
)

sample = test_data_list[0]
input_ids = sample['input_ids'].unsqueeze(0).to(device)
attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
prompt_embeds = sample['prompt_embeds'].unsqueeze(0).to(device)

# --- Generate note ---
with torch.no_grad():
    labels = sample.get('labels', None)
    if labels is not None:
        labels = labels.unsqueeze(0)
        labels_mask = (labels[0] == -100)
        prompt_token_count = torch.sum(labels_mask).item()
    else:
        prompt_token_count = prompt_len
    prompt_input_ids = input_ids[:, :prompt_token_count]
    prompt_attention_mask = attention_mask[:, :prompt_token_count]

    gen_kwargs = {
        "input_ids": prompt_input_ids,
        "attention_mask": prompt_attention_mask,
        "max_new_tokens": 1024,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 1,
        "do_sample": False
    }
    generated_tokens = model.llama.generate(**gen_kwargs)
    pred_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    pred_str = postprocess_generated_note(pred_str)


print("\n===== GENERATED NOTE =====\n")
print(pred_str)
print("\n==========================\n")


