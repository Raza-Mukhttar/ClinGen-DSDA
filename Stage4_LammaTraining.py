# Training Module

import torch
import pandas as pd
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Stage2_BiasedAttentionEncoder import BiasedAttentionEncoder
from Stage3_PromptInjection import PromptProjection
from DataPreprocessing import preprocess_pipeline 
import os
import evaluate
import numpy as np
from transformers import EarlyStoppingCallback
import re
# import setproctitle

# setproctitle.setproctitle("Ready")
import os
import subprocess

# Define the path where your trained NER model should be saved
ner_model_dir = "/outputs/stage1_outputs/final_model/model.safetensors"

# Check if NER model exists, if not, run the training script
if not os.path.exists(ner_model_dir):
    print("NER model not found. Running NER training...")
    subprocess.run(["python", "/Stage1_NER_Module.py"], check=True)
else:
    print("NER model found. Skipping NER training.")



SECTION_HEADERS = [
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "REVIEW OF SYSTEMS",
    "VITALS",
    "RESULTS",
    "INSTRUCTIONS"
]

def extract_sections(text, section_headers=SECTION_HEADERS):
    # Returns dict: {section: content}
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
            # Append content to current section, in case section appears twice
            if sections[curr_section]:
                sections[curr_section] += "\n" + part
            else:
                sections[curr_section] = part
    # Ensure all headers are present
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



# =================== Consistency Loss Helper ===================
import torch.nn.functional as F

def compute_entity_presence(entity_labels, num_entity_types):
    """
    entity_labels: [B, D] or [B, L], values in [0, num_entity_types-1], -100 for ignore
    Returns: [B, num_entity_types] binary presence vector
    """
    mask = (entity_labels != -100)
    masked = torch.where(mask, entity_labels, torch.zeros_like(entity_labels))
    one_hot = F.one_hot(masked, num_classes=num_entity_types)
    # Ignore tokens: set to 0 for all entity types
    one_hot = one_hot * mask.unsqueeze(-1)
    # Binary presence for each entity type in each sample
    presence = (one_hot.sum(dim=1) > 0).float()  # [B, num_entity_types]
    return presence




#========== LoRA Verfication ==========
def print_trainable_parameters(model):
    print("\n==== Trainable parameters (should be LoRA adapters only!) ====")
    total_params = 0
    trainable_params = 0
    for n, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            print(f"Trainable: {n} {p.shape}")
    print(f"Total parameters in model: {total_params:,}")
    print(f"Total *trainable* parameters: {trainable_params:,}")
    print("Trainable ratio: {:.4f}%".format(100 * trainable_params / total_params))

    # PEFT-specific check:
    has_lora = False
    if hasattr(model, "llama") and isinstance(model.llama, PeftModel):
        print("LoRA is ACTIVE! Your model is correctly LoRA-wrapped.\n")
        has_lora = True
    else:
        print("WARNING: LoRA is NOT active! Check model wrapping.\n")
    return has_lora
# =================== Model and Dataset Classes ===================
class PromptedLlamaMultiTask(nn.Module):
    def __init__(self, base_model, prompt_len, hidden_size, ner_label_count, soft_prompt_len=8, lora_cfg=None, consist_loss_fn=None):
        super().__init__()
        self.llama = base_model
        self.prompt_len = prompt_len
        self.hidden_size = hidden_size    
        self.llama_hidden_size = base_model.config.hidden_size   
        self.ner_label_count = ner_label_count
        self.soft_prompt_len = soft_prompt_len
        self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, hidden_size).normal_(0, 0.02))
        self.prompt_proj_to_llama = nn.Linear(self.hidden_size, self.llama_hidden_size)
        if lora_cfg is not None:
            self.llama = get_peft_model(self.llama, lora_cfg)
        self.ner_head = nn.Linear(hidden_size, ner_label_count)
        self.consist_loss_fn = consist_loss_fn or self.default_consist_loss
    def forward(
        self,
        input_ids, attention_mask, prompt_embeds, soft_prompt=None,
        labels=None, ner_context_embeds=None, ner_gold_labels=None, entity_mask=None,
        summary_ner_labels=None  
    ):
        B, L = input_ids.shape
        P = prompt_embeds.shape[1]
        S = self.soft_prompt_len


        target_dtype = self.prompt_proj_to_llama.weight.dtype
        prompt_embeds = prompt_embeds.to(target_dtype)
        if soft_prompt is None:
            soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)
        soft_prompt = soft_prompt.to(target_dtype)
        # Project prompts to llama hidden size
        prompt_embeds_proj = self.prompt_proj_to_llama(prompt_embeds)         
        soft_prompt_proj = self.prompt_proj_to_llama(soft_prompt)             
        input_embeds = self.llama.get_input_embeddings()(input_ids)           

        decoder_embeds = torch.cat([prompt_embeds_proj, soft_prompt_proj, input_embeds], dim=1)  
        full_mask = torch.cat([
            torch.ones((B, P+S), dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask
        ], dim=1)

        # Pad labels for prompt/soft tokens
        labels_padded = labels
        if labels is not None:
            pad_len = decoder_embeds.shape[1] - labels.shape[1]
            if pad_len > 0:
                padding = labels.new_full((labels.shape[0], pad_len), -100)  
                labels_padded = torch.cat([padding, labels], dim=1)          
            else:
                labels_padded = labels

        llama_out = self.llama(
            inputs_embeds=decoder_embeds,
            attention_mask=full_mask,
            labels=labels_padded
        )
        loss = llama_out.loss

        ner_loss = None
        if ner_context_embeds is not None and ner_gold_labels is not None:
            target_dtype = self.ner_head.weight.dtype
            ner_logits = self.ner_head(ner_context_embeds.to(target_dtype))
            ner_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = ner_loss_fct(ner_logits.view(-1, self.ner_label_count), ner_gold_labels.view(-1))

        consist_loss = None
        if (ner_gold_labels is not None) and (summary_ner_labels is not None):
            consist_loss = self.consist_loss_fn(labels, summary_ner_labels, ner_gold_labels, entity_mask)

        lmbd_ner = 0.2
        lmbd_consist = 0.2
        total_loss = loss
        if ner_loss is not None:
            total_loss = total_loss + lmbd_ner * ner_loss
        if consist_loss is not None:
            total_loss = total_loss + lmbd_consist * consist_loss

        return {"loss": total_loss, "main_loss": loss, "ner_loss": ner_loss, "consist_loss": consist_loss}
    
    def default_consist_loss(self, labels, summary_ner_labels, ner_gold_labels, entity_mask):
        """
        Entity consistency loss:
        For each summary token, if its summary_ner_label is a valid entity class (not -100, not O/None),
        check that entity was detected in the dialogue's NER prediction.

        If any summary entity is not found in dialogue, penalize.
        Here, for demo, we just penalize for any entity in summary_ner_labels not present in ner_gold_labels.
        """
        # Get set of predicted entity classes in dialogue (excluding ignore)
        dialogue_entities = set(ner_gold_labels[ner_gold_labels != -100].cpu().numpy().tolist())
        # For each summary token, if its entity label is not in dialogue_entities, count as error
        mask = (summary_ner_labels != -100)
        missing = (mask) & (~torch.isin(summary_ner_labels, torch.tensor(list(dialogue_entities), device=summary_ner_labels.device)))
        consist_loss = missing.float().mean()  
        return consist_loss
    
class MultiTaskSummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]


lr_list = [1e-4, 5e-5, 2e-5, 1e-5]

class StepwiseLRSchedulerCallback(TrainerCallback):
    def __init__(self, patience, lr_list, monitor="eval_loss", mode="min"):
        self.patience = patience
        self.lr_list = lr_list
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.bad_epochs = 0
        self.lr_index = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1] if len(state.log_history) > 0 else {}
        current = logs.get(self.monitor, None)
        if current is None:
            return
        if self.best is None or current < self.best:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.bad_epochs = 0
            self.lr_index += 1
            if self.lr_index < len(self.lr_list):
                new_lr = self.lr_list[self.lr_index]
                print(f"\n[StepwiseLR] Reducing LR to {new_lr} after plateau.")
                for group in kwargs["optimizer"].param_groups:
                    group["lr"] = new_lr
            else:
                print("\n[StepwiseLR] LR schedule exhausted.")


class JointMultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_embeds = inputs.pop('prompt_embeds')
        ner_context_embeds = inputs.get('ner_context_embeds', None)
        ner_gold_labels = inputs.get('ner_gold_labels', None)
        entity_mask = inputs.get('entity_mask', None)
        summary_ner_labels = inputs.get('summary_ner_labels', None)
        output = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            prompt_embeds=prompt_embeds,
            labels=inputs['labels'],
            ner_context_embeds=ner_context_embeds,
            ner_gold_labels=ner_gold_labels,
            entity_mask=entity_mask,
            summary_ner_labels=summary_ner_labels
        )
        loss = output["loss"]
        return (loss, output) if return_outputs else loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        import torch

        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device

            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = inputs.get('labels', None)
            if labels is not None:
                labels_mask = (labels[0] == -100)
                prompt_token_count = torch.sum(labels_mask).item()
            else:
                raise ValueError("labels must be provided for slicing prompt.")

            
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
                "num_beams": 4,
                "do_sample": False
            }
            generated_tokens = model.llama.generate(**gen_kwargs)

            # Loss computation
            prompt_embeds = inputs.get('prompt_embeds', None)
            loss = None
            if labels is not None and input_ids is not None and prompt_embeds is not None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_embeds=prompt_embeds.to(device=device, dtype=model.prompt_proj_to_llama.weight.dtype),
                    labels=labels.to(device),
                    ner_context_embeds=inputs.get("ner_context_embeds", None),
                    ner_gold_labels=inputs.get("ner_gold_labels", None),
                    entity_mask=inputs.get("entity_mask", None),
                    summary_ner_labels=inputs.get("summary_ner_labels", None),
                )
                loss = outputs["loss"].detach()

            return (loss, generated_tokens, labels)


import re

SECTION_HEADERS = [
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "REVIEW OF SYSTEMS",
    "VITALS",
    "PHYSICAL EXAMINATION",
    "RESULTS",
    "ASSESSMENT",
    "PLAN",
    "ASSESSMENT AND PLAN",
    "INSTRUCTIONS"
]

def normalize_section_headers(note_body):
    
    for header in SECTION_HEADERS:
        # Regex: match header (case insensitive), with optional trailing colon and any leading/trailing whitespace
        pattern = rf"(?im)^{header.replace(' ', '[ _]')}[ \t]*:?"
        note_body = re.sub(pattern, header, note_body)
    return note_body

def postprocess_generated_note(generated_text):
    """
    1. Remove everything before [NOTE] (case-insensitive).
    2. Extract content after [NOTE] until the first end tag ([END OF NOTE], [END NOTE], [END INSTRUCTION], [INSTRUCTION]).
    3. Normalize section headers to ALL CAPS (no colon).
    4. Remove leading/trailing whitespace and empty lines.
    """
    # 1. Find [NOTE] (case-insensitive)
    note_match = re.search(r"\[NOTE\]", generated_text, re.IGNORECASE)
    if not note_match:
        return generated_text.strip()
    note_start = note_match.end()
    after_note = generated_text[note_start:]

    # 2. Find first ending tag after [NOTE]
    end_match = re.search(r"\[END OF NOTE\]|\[END NOTE\]|\[END INSTRUCTION\]|\[INSTRUCTION\]", after_note, re.IGNORECASE)
    if end_match:
        note_body = after_note[:end_match.start()]
    else:
        note_body = after_note

    # 3. Normalize section headers
    note_body = normalize_section_headers(note_body)

    # 4. Remove extra blank lines, strip
    note_body = re.sub(r'\n{2,}', '\n\n', note_body)
    return note_body.strip("\n\r\t ")




import sys
sys.path.append('/UniEval')  

from utils import convert_to_json
from metric.evaluator import get_evaluator

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    
    if predictions.ndim == 3:  
        predictions = np.argmax(predictions, axis=-1)

    predictions = predictions.astype(np.int32)
    labels = labels.astype(np.int32)

    # Replace -100 with pad_token_id for decoding
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ---- POSTPROCESS GENERATED NOTES ----
    pred_str = [postprocess_generated_note(p) for p in pred_str]
    label_str = [canonicalize_note(l.strip()) for l in label_str]
    

    
    


    src_list = val_dialogues  
    data = convert_to_json(
        output_list=pred_str,
        src_list=src_list,
        ref_list=label_str
    )
    evaluator = get_evaluator('summarization')
    eval_scores = evaluator.evaluate(data, print_result=True)


    results = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True) 
    bertscore_results = bertscore.compute(predictions=pred_str, references=label_str, lang="en") 


    metrics = {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results["rougeLsum"],
        "bertscore_f1": float(np.mean(bertscore_results["f1"])),
        
    }

    # ---------UniEval---------
    print("Eval scores type:", type(eval_scores))
    print("Eval scores repr:", repr(eval_scores))

    if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
        
        avg_unieval = {}
        keys = eval_scores[0].keys()
        for k in keys:
            avg_unieval["unieval_" + k] = float(np.mean([d[k] for d in eval_scores]))
        metrics.update(avg_unieval)
    elif isinstance(eval_scores, dict):
        metrics.update({f"unieval_{k}": v for k, v in eval_scores.items()})
    else:
        print("UniEval returned unknown type:", type(eval_scores))

    return metrics




# =================== MAIN PIPELINE ===================
if __name__ == "__main__":
    llama_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"  
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(llama_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hidden_size = base_model.config.hidden_size
    prompt_len = 16
    soft_prompt_len = 16
    ner_label_count = 7  
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # ========== DATA LOADING ==========
    train_csv = "/datasets/ClinGen_TrainingSet.csv"
    val_csv = "/datasets/ACI-TaskB-ValidationSet.csv"
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_dialogues = train_df["dialogue"].astype(str).tolist()
    train_notes = train_df["note"].astype(str).tolist()
    val_dialogues = val_df["dialogue"].astype(str).tolist()
    val_notes = val_df["note"].astype(str).tolist()

    # ========== STAGE 1, 2, 3 MODEL LOADING ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_model_dir = "/outputs/stage1_outputs/final_model"
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

    # ========== PREPROCESS (run ONCE after changing data/models) ==========
    if os.path.exists("/outputs/preprocessing_outputs/train_data_list.pt") and os.path.exists("/outputs/preprocessing_outputs/val_data_list.pt"):
        print("Loading preprocessed data...")
        train_data_list = torch.load("/outputs/preprocessing_outputs/train_data_list.pt")
        val_data_list = torch.load("/outputs/preprocessing_outputs/val_data_list.pt")
    else:
        print("Preprocessing training set...")
        train_data_list = preprocess_pipeline(
            train_dialogues, train_notes,
            stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
            tokenizer, device,
            
        )
        print("Preprocessing testing set...")
        val_data_list = preprocess_pipeline(
            val_dialogues, val_notes,
            stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
            tokenizer, device,
            
        )
        torch.save(train_data_list, "/outputs/preprocessing_outputs/train_data_list.pt")
        torch.save(val_data_list, "/outputs/preprocessing_outputs/val_data_list.pt")
        print("Saved train_data_list.pt and val_data_list.pt")

    # ========== TRAINING ==========
    train_dataset = MultiTaskSummarizationDataset(train_data_list)
    val_dataset = MultiTaskSummarizationDataset(val_data_list)
    model = PromptedLlamaMultiTask(
        base_model=base_model,
        prompt_len=prompt_len,
        hidden_size=hidden_size,
        ner_label_count=ner_label_count,
        soft_prompt_len=soft_prompt_len,
        lora_cfg=lora_cfg
    )
    

    print_trainable_parameters(model)

    args = TrainingArguments(
        output_dir="/outputs/model_training_outputs/llama_joint_summ_ckpt",
        deepspeed="/ds_config_zero2.json",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        metric_for_best_model="rouge1",
        load_best_model_at_end=True,
        logging_steps=10,
        learning_rate=5e-5,
        fp16=True,
        report_to="none"
    )
    trainer = JointMultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            StepwiseLRSchedulerCallback(patience=2, lr_list=lr_list, mode="min", monitor="eval_loss")
        ],
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    trainer.save_model("/outputs/model_training_outputs/llama_joint_summ_ckpt/final")
    print("Stage 4 LLM training complete!")



## Comment after training 


eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]

if eval_logs:
    print("\nAverage Evaluation Metrics Across All Epochs:")
    metrics_keys = ['eval_loss', 'eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'eval_rougeLsum', 'eval_bertscore_f1']
    metrics_avgs = {}
    for key in metrics_keys:
        vals = [log[key] for log in eval_logs if key in log]
        if vals:
            avg = np.mean(vals)
            metrics_avgs[key] = avg
            print(f"{key}: {avg:.4f}")
    # Optionally, print per-epoch values:
    print("\nPer-Epoch Evaluation Metrics:")
    for i, log in enumerate(eval_logs):
        print(f"Epoch {log.get('epoch', i+1)}: " + ", ".join([f"{k}: {log[k]:.4f}" for k in metrics_keys if k in log]))
else:
    print("No evaluation logs found.")



