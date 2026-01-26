# # ------------------------------
# # Keep TF quiet & off the GPUs (for BLEURT)
# # ------------------------------
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# try:
#     import tensorflow as tf
#     try:
#         tf.config.set_visible_devices([], 'GPU')  # disable all TF GPUs
#     except Exception:
#         pass
#     try:
#         tf.config.threading.set_intra_op_parallelism_threads(1)
#         tf.config.threading.set_inter_op_parallelism_threads(1)
#     except Exception:
#         pass
# except Exception:
#     pass

# # ------------------------------


# import torch
# import pandas as pd
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification, TrainerCallback
# from peft import LoraConfig, get_peft_model, PeftModel
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from Stage2 import BiasedAttentionEncoder
# from Stage3 import PromptProjection
# from DataPreprocessing import preprocess_pipeline 
# import os
# import evaluate
# import numpy as np
# from transformers import EarlyStoppingCallback
# import re
# # import setproctitle

# # setproctitle.setproctitle("Ready")
# import os
# import subprocess

# # Define the path where your trained NER model should be saved
# ner_model_dir = "/home/raza/EMRV1/EMRV2/outputs/stage1_outputs/final_model/model.safetensors"

# # Check if NER model exists, if not, run the training script
# if not os.path.exists(ner_model_dir):
#     print("NER model not found. Running NER training...")
#     subprocess.run(["python", "/home/raza/EMRV1/EMRV2/Stage1.py"], check=True)
# else:
#     print("NER model found. Skipping NER training.")



# SECTION_HEADERS = [
#     "CHIEF COMPLAINT",
#     "HISTORY OF PRESENT ILLNESS",
#     "REVIEW OF SYSTEMS",
#     "VITALS",
#     "RESULTS",
#     "INSTRUCTIONS"
# ]

# def extract_sections(text, section_headers=SECTION_HEADERS):
#     # Returns dict: {section: content}
#     sections = {}
#     pattern = '|'.join([re.escape(h) for h in section_headers])
#     parts = re.split(f"({pattern})", text)
#     curr_section = None
#     for part in parts:
#         part = part.strip()
#         if part in section_headers:
#             curr_section = part
#             sections[curr_section] = ""
#         elif curr_section and part:
#             # Append content to current section, in case section appears twice
#             if sections[curr_section]:
#                 sections[curr_section] += "\n" + part
#             else:
#                 sections[curr_section] = part
#     # Ensure all headers are present
#     for h in section_headers:
#         if h not in sections:
#             sections[h] = ""
#     return sections



# def canonicalize_note(text, section_headers=SECTION_HEADERS):
#     sections = extract_sections(text, section_headers)
#     canonical = ""
#     for h in section_headers:
#         canonical += h + "\n\n" + sections[h].strip() + "\n\n"
#     return canonical.strip()



# # =================== Consistency Loss Helper ===================
# import torch.nn.functional as F

# def compute_entity_presence(entity_labels, num_entity_types):
#     """
#     entity_labels: [B, D] or [B, L], values in [0, num_entity_types-1], -100 for ignore
#     Returns: [B, num_entity_types] binary presence vector
#     """
#     mask = (entity_labels != -100)
#     masked = torch.where(mask, entity_labels, torch.zeros_like(entity_labels))
#     one_hot = F.one_hot(masked, num_classes=num_entity_types)
#     # Ignore tokens: set to 0 for all entity types
#     one_hot = one_hot * mask.unsqueeze(-1)
#     # Binary presence for each entity type in each sample
#     presence = (one_hot.sum(dim=1) > 0).float()  # [B, num_entity_types]
#     return presence




# #========== LoRA Verfication ==========
# def print_trainable_parameters(model):
#     print("\n==== Trainable parameters (should be LoRA adapters only!) ====")
#     total_params = 0
#     trainable_params = 0
#     for n, p in model.named_parameters():
#         total_params += p.numel()
#         if p.requires_grad:
#             trainable_params += p.numel()
#             print(f"Trainable: {n} {p.shape}")
#     print(f"Total parameters in model: {total_params:,}")
#     print(f"Total *trainable* parameters: {trainable_params:,}")
#     print("Trainable ratio: {:.4f}%".format(100 * trainable_params / total_params))

#     # PEFT-specific check:
#     has_lora = False
#     if hasattr(model, "llama") and isinstance(model.llama, PeftModel):
#         print("LoRA is ACTIVE! Your model is correctly LoRA-wrapped.\n")
#         has_lora = True
#     else:
#         print("WARNING: LoRA is NOT active! Check model wrapping.\n")
#     return has_lora
# # =================== Model and Dataset Classes ===================
# class PromptedLlamaMultiTask(nn.Module):
#     def __init__(self, base_model, prompt_len, hidden_size, ner_label_count, soft_prompt_len=8, lora_cfg=None, consist_loss_fn=None):
#         super().__init__()
#         self.llama = base_model
#         self.prompt_len = prompt_len
#         self.hidden_size = hidden_size    
#         self.llama_hidden_size = base_model.config.hidden_size   
#         self.ner_label_count = ner_label_count
#         self.soft_prompt_len = soft_prompt_len
#         self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, hidden_size).normal_(0, 0.02))
#         self.prompt_proj_to_llama = nn.Linear(self.hidden_size, self.llama_hidden_size)
#         if lora_cfg is not None:
#             self.llama = get_peft_model(self.llama, lora_cfg)
#         self.ner_head = nn.Linear(hidden_size, ner_label_count)
#         self.consist_loss_fn = consist_loss_fn or self.default_consist_loss
#     def forward(
#         self,
#         input_ids, attention_mask, prompt_embeds, soft_prompt=None,
#         labels=None, ner_context_embeds=None, ner_gold_labels=None, entity_mask=None,
#         summary_ner_labels=None  
#     ):
#         B, L = input_ids.shape
#         P = prompt_embeds.shape[1]
#         S = self.soft_prompt_len


#         target_dtype = self.prompt_proj_to_llama.weight.dtype
#         prompt_embeds = prompt_embeds.to(target_dtype)
#         if soft_prompt is None:
#             soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)
#         soft_prompt = soft_prompt.to(target_dtype)
#         # Project prompts to llama hidden size
#         prompt_embeds_proj = self.prompt_proj_to_llama(prompt_embeds)         
#         soft_prompt_proj = self.prompt_proj_to_llama(soft_prompt)             
#         input_embeds = self.llama.get_input_embeddings()(input_ids)           

#         decoder_embeds = torch.cat([prompt_embeds_proj, soft_prompt_proj, input_embeds], dim=1)  
#         full_mask = torch.cat([
#             torch.ones((B, P+S), dtype=attention_mask.dtype, device=attention_mask.device),
#             attention_mask
#         ], dim=1)

#         # Pad labels for prompt/soft tokens
#         labels_padded = labels
#         if labels is not None:
#             pad_len = decoder_embeds.shape[1] - labels.shape[1]
#             if pad_len > 0:
#                 padding = labels.new_full((labels.shape[0], pad_len), -100)  
#                 labels_padded = torch.cat([padding, labels], dim=1)          
#             else:
#                 labels_padded = labels

#         llama_out = self.llama(
#             inputs_embeds=decoder_embeds,
#             attention_mask=full_mask,
#             labels=labels_padded
#         )
#         loss = llama_out.loss

#         ner_loss = None
#         if ner_context_embeds is not None and ner_gold_labels is not None:
#             target_dtype = self.ner_head.weight.dtype
#             ner_logits = self.ner_head(ner_context_embeds.to(target_dtype))
#             ner_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#             ner_loss = ner_loss_fct(ner_logits.view(-1, self.ner_label_count), ner_gold_labels.view(-1))

#         consist_loss = None
#         if (ner_gold_labels is not None) and (summary_ner_labels is not None):
#             consist_loss = self.consist_loss_fn(labels, summary_ner_labels, ner_gold_labels, entity_mask)

#         lmbd_ner = 0.2
#         lmbd_consist = 0.2
#         total_loss = loss
#         if ner_loss is not None:
#             total_loss = total_loss + lmbd_ner * ner_loss
#         if consist_loss is not None:
#             total_loss = total_loss + lmbd_consist * consist_loss

#         return {"loss": total_loss, "main_loss": loss, "ner_loss": ner_loss, "consist_loss": consist_loss}
    
#     def default_consist_loss(self, labels, summary_ner_labels, ner_gold_labels, entity_mask):
#         """
#         Entity consistency loss:
#         For each summary token, if its summary_ner_label is a valid entity class (not -100, not O/None),
#         check that entity was detected in the dialogue's NER prediction.

#         If any summary entity is not found in dialogue, penalize.
#         Here, for demo, we just penalize for any entity in summary_ner_labels not present in ner_gold_labels.
#         """
#         # Get set of predicted entity classes in dialogue (excluding ignore)
#         dialogue_entities = set(ner_gold_labels[ner_gold_labels != -100].cpu().numpy().tolist())
#         # For each summary token, if its entity label is not in dialogue_entities, count as error
#         mask = (summary_ner_labels != -100)
#         missing = (mask) & (~torch.isin(summary_ner_labels, torch.tensor(list(dialogue_entities), device=summary_ner_labels.device)))
#         consist_loss = missing.float().mean()  
#         return consist_loss
    
# class MultiTaskSummarizationDataset(torch.utils.data.Dataset):
#     def __init__(self, data_list):
#         self.data_list = data_list
#     def __len__(self):
#         return len(self.data_list)
#     def __getitem__(self, idx):
#         return self.data_list[idx]


# lr_list = [1e-4, 5e-5, 2e-5, 1e-5]

# class StepwiseLRSchedulerCallback(TrainerCallback):
#     def __init__(self, patience, lr_list, monitor="eval_loss", mode="min"):
#         self.patience = patience
#         self.lr_list = lr_list
#         self.monitor = monitor
#         self.mode = mode
#         self.best = None
#         self.bad_epochs = 0
#         self.lr_index = 0

#     def on_epoch_end(self, args, state, control, **kwargs):
#         logs = state.log_history[-1] if len(state.log_history) > 0 else {}
#         current = logs.get(self.monitor, None)
#         if current is None:
#             return
#         if self.best is None or current < self.best:
#             self.best = current
#             self.bad_epochs = 0
#         else:
#             self.bad_epochs += 1
#         if self.bad_epochs >= self.patience:
#             self.bad_epochs = 0
#             self.lr_index += 1
#             if self.lr_index < len(self.lr_list):
#                 new_lr = self.lr_list[self.lr_index]
#                 print(f"\n[StepwiseLR] Reducing LR to {new_lr} after plateau.")
#                 for group in kwargs["optimizer"].param_groups:
#                     group["lr"] = new_lr
#             else:
#                 print("\n[StepwiseLR] LR schedule exhausted.")


# class JointMultiTaskTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         prompt_embeds = inputs.pop('prompt_embeds')
#         ner_context_embeds = inputs.get('ner_context_embeds', None)
#         ner_gold_labels = inputs.get('ner_gold_labels', None)
#         entity_mask = inputs.get('entity_mask', None)
#         summary_ner_labels = inputs.get('summary_ner_labels', None)
#         output = model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             prompt_embeds=prompt_embeds,
#             labels=inputs['labels'],
#             ner_context_embeds=ner_context_embeds,
#             ner_gold_labels=ner_gold_labels,
#             entity_mask=entity_mask,
#             summary_ner_labels=summary_ner_labels
#         )
#         loss = output["loss"]
#         return (loss, output) if return_outputs else loss


#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
#         import torch

#         model.eval()
#         with torch.no_grad():
#             device = next(model.parameters()).device

            
#             input_ids = inputs['input_ids'].to(device)
#             attention_mask = inputs['attention_mask'].to(device)
#             labels = inputs.get('labels', None)
#             if labels is not None:
#                 labels_mask = (labels[0] == -100)
#                 prompt_token_count = torch.sum(labels_mask).item()
#             else:
#                 raise ValueError("labels must be provided for slicing prompt.")

            
#             prompt_input_ids = input_ids[:, :prompt_token_count]
#             prompt_attention_mask = attention_mask[:, :prompt_token_count]

#             gen_kwargs = {
#                 "input_ids": prompt_input_ids,
#                 "attention_mask": prompt_attention_mask,
#                 "max_new_tokens": 1024,
#                 "min_new_tokens": 512,
#                 # "early_stopping": True,
#                 "temperature": 0.7,
#                 "top_p": 0.9,
#                 # "use_cache": True,
#                 "eos_token_id": tokenizer.eos_token_id,
#                 "pad_token_id": tokenizer.pad_token_id,
#                 "repetition_penalty": 1.2,
#                 "no_repeat_ngram_size": 3,
#                 "num_beams": 4,
#                 "do_sample": True
#             }
#             generated_tokens = model.llama.generate(**gen_kwargs)

#             # Loss computation
#             prompt_embeds = inputs.get('prompt_embeds', None)
#             loss = None
#             if labels is not None and input_ids is not None and prompt_embeds is not None:
#                 outputs = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     prompt_embeds=prompt_embeds.to(device=device, dtype=model.prompt_proj_to_llama.weight.dtype),
#                     labels=labels.to(device),
#                     ner_context_embeds=inputs.get("ner_context_embeds", None),
#                     ner_gold_labels=inputs.get("ner_gold_labels", None),
#                     entity_mask=inputs.get("entity_mask", None),
#                     summary_ner_labels=inputs.get("summary_ner_labels", None),
#                 )
#                 loss = outputs["loss"].detach()

#             return (loss, generated_tokens, labels)


# import re

# SECTION_HEADERS = [
#     "CHIEF COMPLAINT",
#     "HISTORY OF PRESENT ILLNESS",
#     "REVIEW OF SYSTEMS",
#     "VITALS",
#     "PHYSICAL EXAMINATION",
#     "RESULTS",
#     "ASSESSMENT",
#     "PLAN",
#     "ASSESSMENT AND PLAN",
#     "INSTRUCTIONS"
# ]

# def normalize_section_headers(note_body):
    
#     for header in SECTION_HEADERS:
#         # Regex: match header (case insensitive), with optional trailing colon and any leading/trailing whitespace
#         pattern = rf"(?im)^{header.replace(' ', '[ _]')}[ \t]*:?"
#         note_body = re.sub(pattern, header, note_body)
#     return note_body

# def postprocess_generated_note(generated_text):
#     """
#     1. Remove everything before [NOTE] (case-insensitive).
#     2. Extract content after [NOTE] until the first end tag ([END OF NOTE], [END NOTE], [END INSTRUCTION], [INSTRUCTION]).
#     3. Normalize section headers to ALL CAPS (no colon).
#     4. Remove leading/trailing whitespace and empty lines.
#     """
#     # 1. Find [NOTE] (case-insensitive)
#     note_match = re.search(r"\[NOTE\]", generated_text, re.IGNORECASE)
#     if not note_match:
#         return generated_text.strip()
#     note_start = note_match.end()
#     after_note = generated_text[note_start:]

#     # 2. Find first ending tag after [NOTE]
#     end_match = re.search(r"\[END OF NOTE\]|\[END NOTE\]|\[END INSTRUCTION\]|\[INSTRUCTION\]", after_note, re.IGNORECASE)
#     if end_match:
#         note_body = after_note[:end_match.start()]
#     else:
#         note_body = after_note

#     # 3. Normalize section headers
#     note_body = normalize_section_headers(note_body)

#     # 4. Remove extra blank lines, strip
#     note_body = re.sub(r'\n{2,}', '\n\n', note_body)
#     return note_body.strip("\n\r\t ")




# import sys
# sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')  

# from utils import convert_to_json
# from metric.evaluator import get_evaluator

# rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")


# meteor = evaluate.load("meteor")
# sacrebleu = evaluate.load("sacrebleu")
# bleurt = evaluate.load("bleurt", config_name="bleurt-20")


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred

    
#     if isinstance(predictions, tuple):
#         predictions = predictions[0]
#     if isinstance(labels, tuple):
#         labels = labels[0]

    
#     if predictions.ndim == 3:  
#         predictions = np.argmax(predictions, axis=-1)

#     predictions = predictions.astype(np.int32)
#     labels = labels.astype(np.int32)

#     # Replace -100 with pad_token_id for decoding
#     labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
#     predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)

#     pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # ---- POSTPROCESS GENERATED NOTES ----
#     pred_str = [postprocess_generated_note(p) for p in pred_str]
#     label_str = [canonicalize_note(l.strip()) for l in label_str]
    

    
    


#     src_list = val_dialogues  
#     data = convert_to_json(
#         output_list=pred_str,
#         src_list=src_list,
#         ref_list=label_str
#     )
#     evaluator = get_evaluator('summarization')
#     eval_scores = evaluator.evaluate(data, print_result=True)


#     results = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True) 
#     bertscore_results = bertscore.compute(predictions=pred_str, references=label_str, lang="en") 


#     # --- Additional metrics ---
#     # METEOR: list[str], list[str]
#     meteor_result = meteor.compute(predictions=pred_str, references=label_str)
#     # SACREBLEU: list[str], list[list[str]]
#     sacrebleu_result = sacrebleu.compute(predictions=pred_str, references=[[r] for r in label_str])
#     # BLEURT: list[str], list[str]  (runs on CPU via evaluate)
#     try:
#         bleurt_result = bleurt.compute(predictions=pred_str, references=label_str)
#         bleurt_mean = float(np.mean(bleurt_result["scores"]))
#     except Exception:
#         bleurt_mean = float("nan")

#     metrics = {
#         "rouge1": results["rouge1"],
#         "rouge2": results["rouge2"],
#         "rougeL": results["rougeL"],
#         "rougeLsum": results["rougeLsum"],
#         "bertscore_f1": float(np.mean(bertscore_results["f1"])),
#         "meteor": float(meteor_result["meteor"]),
#         "bleu": float(sacrebleu_result["score"]),
#         "bleurt": bleurt_mean,
        
#     }

#     # ---------UniEval---------
#     print("Eval scores type:", type(eval_scores))
#     print("Eval scores repr:", repr(eval_scores))

#     if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
        
#         avg_unieval = {}
#         keys = eval_scores[0].keys()
#         for k in keys:
#             avg_unieval["unieval_" + k] = float(np.mean([d[k] for d in eval_scores]))
#         metrics.update(avg_unieval)
#     elif isinstance(eval_scores, dict):
#         metrics.update({f"unieval_{k}": v for k, v in eval_scores.items()})
#     else:
#         print("UniEval returned unknown type:", type(eval_scores))

#     return metrics




# # =================== MAIN PIPELINE ===================
# if __name__ == "__main__":
#     # llama_model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  
#     # llama_model_path = "Qwen/Qwen2-7B-Instruct" 
#     # llama_model_path = "mistralai/Mistral-7B-v0.3"
#     # llama_model_path = "Qwen/Qwen3-14B"
#     llama_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
#     # tokenizer = AutoTokenizer.from_pretrained(
#     #     llama_model_path,
#     #     use_fast=False,            # <— key change
#     #     trust_remote_code=True
#     #     )
#     # base_model = AutoModelForCausalLM.from_pretrained(
#     #     llama_model_path,
#     #     trust_remote_code=True,
#     #     torch_dtype=torch.bfloat16  # or fp16 if you prefer
#     #     )

#     # if tokenizer.pad_token is None:
#     #     tokenizer.pad_token = tokenizer.eos_token


#     tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=True)
#     base_model = AutoModelForCausalLM.from_pretrained(llama_model_path)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     hidden_size = base_model.config.hidden_size
#     prompt_len = 16
#     soft_prompt_len = 16
#     ner_label_count = 7  
#     lora_cfg = LoraConfig(
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=["q_proj", "v_proj"]
#         # target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] # for qwen3-8B
#     )

#     # ========== DATA LOADING ==========
#     train_csv = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-TrainingSet.csv"
#     val_csv = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"
#     train_df = pd.read_csv(train_csv)
#     val_df = pd.read_csv(val_csv)
#     train_dialogues = train_df["dialogue"].astype(str).tolist()
#     train_notes = train_df["note"].astype(str).tolist()
#     val_dialogues = val_df["dialogue"].astype(str).tolist()
#     val_notes = val_df["note"].astype(str).tolist()

#     # ========== STAGE 1, 2, 3 MODEL LOADING ==========
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     stage1_model_dir = "/home/raza/EMRV1/EMRV2/outputs/stage1_outputs/final_model"
#     stage1_model = AutoModelForTokenClassification.from_pretrained(stage1_model_dir).to(device)
#     stage1_tokenizer = AutoTokenizer.from_pretrained(stage1_model_dir)
#     backbone_name = "emilyalsentzer/Bio_ClinicalBERT"
#     hidden_size = stage1_model.config.hidden_size
#     entity_label_count = stage1_model.config.num_labels
#     stage2_encoder = BiasedAttentionEncoder(
#         backbone_name=backbone_name,
#         hidden_size=hidden_size,
#         entity_label_count=entity_label_count,
#         bias_strength=3.0
#     ).to(device)
#     stage3_proj = PromptProjection(
#         hidden_size=hidden_size,
#         prompt_len=16,
#         mode="attentive"
#     )

#     # ========== PREPROCESS (run ONCE after changing data/models) ==========
#     if os.path.exists("/home/raza/EMRV/EMRV2/outputs/preprocessing_outputs/train_data_list.pt") and os.path.exists("/home/raza/EMRV/EMRV2/outputs/preprocessing_outputs/val_data_list.pt"):
#         print("Loading preprocessed data...")
#         train_data_list = torch.load("/home/raza/EMRV/EMRV2/outputs/preprocessing_outputs/train_data_list.pt")
#         val_data_list = torch.load("/home/raza/EMRV/EMRV2/outputs/preprocessing_outputs/val_data_list.pt")

#     # if os.path.exists("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_qwen/train_data_list.pt") and os.path.exists("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_qwen/val_data_list.pt"):
#     #     print("Loading preprocessed data...")
#     #     train_data_list = torch.load("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_qwen/train_data_list.pt")
#     #     val_data_list = torch.load("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_qwen/val_data_list.pt")

#     # if os.path.exists("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_mistral/trainq_data_list.pt") and os.path.exists("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_mistral/valq_data_list.pt"):
#     #     print("Loading preprocessed data...")
#     #     train_data_list = torch.load("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_mistral/trainq_data_list.pt")
#     #     val_data_list = torch.load("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_mistral/valq_data_list.pt")

#     # if os.path.exists("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/train70b_data_list.pt") and os.path.exists("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/val70b_data_list.pt"):
#     #     print("Loading preprocessed data...")
#     #     train_data_list = torch.load("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/train70b_data_list.pt")
#     #     val_data_list = torch.load("/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/val70b_data_list.pt")
#     else:
#         print("Preprocessing training set...")
#         train_data_list = preprocess_pipeline(
#             train_dialogues, train_notes,
#             stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
#             tokenizer, device,
            
#         )
#         print("Preprocessing testing set...")
#         val_data_list = preprocess_pipeline(
#             val_dialogues, val_notes,
#             stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
#             tokenizer, device,
            
#         )
#         torch.save(train_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/train_data_list.pt")
#         torch.save(val_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/val_data_list.pt")

#         # torch.save(train_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_qwen/train_data_list.pt")
#         # torch.save(val_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_qwen/val_data_list.pt")

#         # torch.save(train_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_mistral/trainq_data_list.pt")
#         # torch.save(val_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs_mistral/valq_data_list.pt")

#         # torch.save(train_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/train70b_data_list.pt")
#         # torch.save(val_data_list, "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/val70b_data_list.pt")
#         print("Saved train_data_list.pt and val_data_list.pt")

#     # ========== TRAINING ==========
#     train_dataset = MultiTaskSummarizationDataset(train_data_list)
#     val_dataset = MultiTaskSummarizationDataset(val_data_list)
#     model = PromptedLlamaMultiTask(
#         base_model=base_model,
#         prompt_len=prompt_len,
#         hidden_size=hidden_size,
#         ner_label_count=ner_label_count,
#         soft_prompt_len=soft_prompt_len,
#         lora_cfg=lora_cfg
#     )
    

#     print_trainable_parameters(model)

#     args = TrainingArguments(
#         output_dir="/home/raza/EMRV/EMRV2/outputs/model_training_outputs/llama_joint_summ_ckpt",
#         # output_dir="/home/raza/EMRV1/EMRV2/outputs/model_training_outputs/llamaqwen_joint_summ_ckpt",
#         # output_dir="/home/raza/EMRV1/EMRV2/outputs/model_training_outputs/mistralq_joint_summ_ckpt",
#         deepspeed="/home/raza/EMRV1/EMRV2/ds_config_zero2.json",
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=2,
#         num_train_epochs=3,
#         save_total_limit=1,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         warmup_ratio=0.1,
#         # gradient_checkpointing=True,
#         lr_scheduler_type="cosine",
#         metric_for_best_model="rouge1",
#         load_best_model_at_end=True,
#         logging_steps=10,
#         learning_rate=5e-5,
#         fp16=True,
#         report_to="none"
#     )
#     trainer = JointMultiTaskTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         callbacks=[
#             EarlyStoppingCallback(early_stopping_patience=3),
#             StepwiseLRSchedulerCallback(patience=2, lr_list=lr_list, mode="min", monitor="eval_loss")
#         ],
#         compute_metrics=compute_metrics
#     )
#     trainer.train()
    
#     trainer.save_model("/home/raza/EMRV1/EMRV2/outputs/model_training_outputs/llama70b_joint_summ_ckpt/final")
#     print("Stage 4 LLM training complete!")



# # Comment after training 


# eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]

# if eval_logs:
#     print("\nAverage Evaluation Metrics mistral-7B:")
#     metrics_keys = ['eval_loss', 'eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'eval_rougeLsum', 'eval_bertscore_f1', 'eval_meteor', 'eval_bleu', 'eval_bleurt']
#     metrics_avgs = {}
#     for key in metrics_keys:
#         vals = [log[key] for log in eval_logs if key in log]
#         if vals:
#             avg = np.mean(vals)
#             metrics_avgs[key] = avg
#             print(f"{key}: {avg:.4f}")
#     # Optionally, print per-epoch values:
#     print("\nPer-Epoch Evaluation Metrics:")
#     for i, log in enumerate(eval_logs):
#         print(f"Epoch {log.get('epoch', i+1)}: " + ", ".join([f"{k}: {log[k]:.4f}" for k in metrics_keys if k in log]))
# else:
#     print("No evaluation logs found.")















# Stage4_qlora.py   CUDA_VISIBLE_DEVICES=2 TRAIN_MAX_LEN=512 EVAL_MAX_LEN=512 deepspeed --num_gpus=1 Stage4.py
# QLoRA version of your Stage4 script – minimal changes, same pipeline.
import os
# Make only GPUs 1,2,3,4 visible to this process (logical 0..3)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")   # often avoids small extra buffers
os.environ.setdefault("NCCL_SHM_DISABLE", "1")   # reduces shared-mem usage
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
# ------------------------------
# Keep TF quiet & off the GPUs (for BLEURT)
# ------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import tensorflow as tf
    try: tf.config.set_visible_devices([], 'GPU')
    except Exception: pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception: pass
except Exception:
    pass

# ------------------------------

import json, re, sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import evaluate

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification,
    TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from peft import prepare_model_for_kbit_training

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Stage2 import BiasedAttentionEncoder
from Stage3 import PromptProjection
from EMRV1.EMRV2.Endocrinology.DataPreprocessingEndo import preprocess_pipeline

# ------------------------------ Stage1 bootstrap ------------------------------
ner_model_dir = "/home/raza/EMRV1/EMRV2/outputs/stage1_outputs/final_model/model.safetensors"
if not os.path.exists(ner_model_dir):
    import subprocess
    print("NER model not found. Running NER training...")
    subprocess.run(["python", "/home/raza/EMRV1/EMRV2/Stage1.py"], check=True)
else:
    print("NER model found. Skipping NER training.")

# ------------------------------ Utilities ------------------------------
SECTION_HEADERS = ["CHIEF COMPLAINT","HISTORY OF PRESENT ILLNESS","REVIEW OF SYSTEMS","VITALS","RESULTS","INSTRUCTIONS"]
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

def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal *trainable* parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100.0*trainable_params/total_params:.4f}%")
    if hasattr(model, "llama") and isinstance(model.llama, PeftModel):
        print("LoRA is ACTIVE! Your model is correctly LoRA-wrapped.\n")
    else:
        print("WARNING: LoRA is NOT active! Check model wrapping.\n")

# ------------------------------ Model wrapper ------------------------------
class PromptedLlamaMultiTask(nn.Module):
    def __init__(self, base_model, prompt_len, encoder_hidden_size, ner_label_count,
                 soft_prompt_len=8, lora_cfg=None, consist_loss_fn=None):
        super().__init__()
        self.llama = base_model
        self.prompt_len = prompt_len
        self.encoder_hidden_size = encoder_hidden_size         # e.g., 768 (ClinicalBERT)
        self.llama_hidden_size = base_model.config.hidden_size # e.g., 8192 (Llama-3-70B)
        self.ner_label_count = ner_label_count
        self.soft_prompt_len = soft_prompt_len

        self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, self.encoder_hidden_size).normal_(0, 0.02))
        self.prompt_proj_to_llama = nn.Linear(self.encoder_hidden_size, self.llama_hidden_size)

        if lora_cfg is not None:
            self.llama = get_peft_model(self.llama, lora_cfg)

        self.ner_head = nn.Linear(self.encoder_hidden_size, ner_label_count)
        self.consist_loss_fn = consist_loss_fn or self.default_consist_loss

    # ---- Let HF enable GC on our wrapped model
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        if hasattr(self.llama, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs is not None:
                    self.llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                else:
                    self.llama.gradient_checkpointing_enable()
            except TypeError:
                self.llama.gradient_checkpointing_enable()
        try: self.llama.config.use_cache = False
        except Exception: pass
        for m in self.modules():
            if hasattr(m, "gradient_checkpointing"):
                setattr(m, "gradient_checkpointing", True)

    def gradient_checkpointing_disable(self):
        if hasattr(self.llama, "gradient_checkpointing_disable"):
            try: self.llama.gradient_checkpointing_disable()
            except TypeError: self.llama.gradient_checkpointing_disable()
        for m in self.modules():
            if hasattr(m, "gradient_checkpointing"):
                setattr(m, "gradient_checkpointing", False)

    def forward(self,
        input_ids, attention_mask, prompt_embeds, soft_prompt=None,
        labels=None, ner_context_embeds=None, ner_gold_labels=None, entity_mask=None,
        summary_ner_labels=None
    ):
        B, L = input_ids.shape
        P = prompt_embeds.shape[1]
        S = self.soft_prompt_len

        target_dtype = getattr(self.llama, "dtype", self.prompt_proj_to_llama.weight.dtype)
        prompt_embeds = prompt_embeds.to(target_dtype)
        if soft_prompt is None:
            soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(target_dtype)

        prompt_embeds_proj = self.prompt_proj_to_llama(prompt_embeds)   # [B,P,enc] -> [B,P,llm]
        soft_prompt_proj   = self.prompt_proj_to_llama(soft_prompt)     # [B,S,enc] -> [B,S,llm]
        input_embeds       = self.llama.get_input_embeddings()(input_ids)

        decoder_embeds = torch.cat([prompt_embeds_proj, soft_prompt_proj, input_embeds], dim=1)
        full_mask = torch.cat([
            torch.ones((B, P+S), dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask
        ], dim=1)

        labels_padded = labels
        if labels is not None:
            pad_len = decoder_embeds.shape[1] - labels.shape[1]
            if pad_len > 0:
                labels_padded = torch.cat([labels.new_full((labels.shape[0], pad_len), -100), labels], dim=1)

        llama_out = self.llama(inputs_embeds=decoder_embeds, attention_mask=full_mask, labels=labels_padded)
        loss = llama_out.loss

        ner_loss = None
        if ner_context_embeds is not None and ner_gold_labels is not None:
            ner_logits = self.ner_head(ner_context_embeds.to(self.ner_head.weight.dtype))
            ner_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = ner_loss_fct(ner_logits.view(-1, self.ner_label_count), ner_gold_labels.view(-1))

        consist_loss = None
        if (ner_gold_labels is not None) and (summary_ner_labels is not None):
            consist_loss = self.default_consist_loss(labels, summary_ner_labels, ner_gold_labels, entity_mask)

        total_loss = loss
        if ner_loss is not None: total_loss = total_loss + 0.2 * ner_loss
        if consist_loss is not None: total_loss = total_loss + 0.2 * consist_loss
        return {"loss": total_loss, "main_loss": loss, "ner_loss": ner_loss, "consist_loss": consist_loss}

    def default_consist_loss(self, labels, summary_ner_labels, ner_gold_labels, entity_mask):
        dialogue_entities = set(ner_gold_labels[ner_gold_labels != -100].detach().cpu().numpy().tolist())
        mask = (summary_ner_labels != -100)
        if len(dialogue_entities) == 0: return (mask.float().mean()) * 0.0
        missing = (mask) & (~torch.isin(summary_ner_labels, torch.tensor(list(dialogue_entities), device=summary_ner_labels.device)))
        return missing.float().mean()

class MultiTaskSummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

# ------------------------------ Memory-safe collator (truncate) ------------------------------
class DataCollatorTruncate:
    def __init__(self, train_max_len=2048, eval_max_len=1024, for_eval=False):
        self.max_len = eval_max_len if for_eval else train_max_len
        self.for_eval = for_eval

    def __call__(self, batch):
        # batch is a list of dicts produced by your preprocess_pipeline
        out = {}
        keys = batch[0].keys()
        for k in keys:
            if k in ("input_ids", "attention_mask", "labels"):
                seqs = []
                for b in batch:
                    v = b[k]
                    if isinstance(v, torch.Tensor):
                        # avoid extra copy & warning
                        seqs.append(v)  # if you must copy: v.detach().clone()
                    else:
                        seqs.append(torch.tensor(v))
                pad_val = 0 if k != "labels" else -100
                tens = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_val)
                if tens.size(1) > self.max_len:
                    tens = tens[:, -self.max_len:]   # tail-truncate
                out[k] = tens
            elif k in ("prompt_embeds","ner_context_embeds","ner_gold_labels","entity_mask","summary_ner_labels"):
                # Leave as-is, stack if tensor-like
                vals = [b[k] for b in batch]
                if isinstance(vals[0], torch.Tensor):
                    out[k] = torch.stack(vals, dim=0)
                else:
                    out[k] = vals
            else:
                out[k] = [b[k] for b in batch]
        return out

# ------------------------------ LR schedule ------------------------------
lr_list = [1e-4, 5e-5, 2e-5, 1e-5]
class StepwiseLRSchedulerCallback(TrainerCallback):
    def __init__(self, patience, lr_list, monitor="eval_loss", mode="min"):
        self.patience = patience; self.lr_list = lr_list
        self.monitor = monitor; self.mode = mode
        self.best = None; self.bad_epochs = 0; self.lr_index = 0
    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history[-1] if len(state.log_history) > 0 else {}
        current = logs.get(self.monitor, None)
        if current is None: return
        if self.best is None or current < self.best:
            self.best = current; self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.bad_epochs = 0; self.lr_index += 1
            if self.lr_index < len(self.lr_list):
                new_lr = self.lr_list[self.lr_index]
                print(f"\n[StepwiseLR] Reducing LR to {new_lr} after plateau.")
                for group in kwargs["optimizer"].param_groups: group["lr"] = new_lr
            else:
                print("\n[StepwiseLR] LR schedule exhausted.")

# ------------------------------ Trainer ------------------------------
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

    # === REPLACE THIS METHOD ===
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        import torch
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            target_dtype = model.prompt_proj_to_llama.weight.dtype

            # batch fields
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            prompt_embeds = inputs['prompt_embeds'].to(device, dtype=target_dtype)
            labels = inputs.get('labels', None)

            B = input_ids.size(0)

            # 1) project Stage3 prompt
            prompt_proj = model.prompt_proj_to_llama(prompt_embeds)  # [B, P, H_llama]

            # 2) soft prompt
            soft_prompt = model.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(device, dtype=target_dtype)
            soft_proj = model.prompt_proj_to_llama(soft_prompt)      # [B, S, H_llama]

            # 3) dialogue tokens -> embeddings
            in_embeds = model.llama.get_input_embeddings()(input_ids)  # [B, L, H_llama]

            # 4) final prefix+dialogue embeddings & mask
            gen_inputs_embeds = torch.cat([prompt_proj, soft_proj, in_embeds], dim=1)
            prefix_len = prompt_proj.size(1) + soft_proj.size(1)
            gen_attn = torch.cat(
                [torch.ones((B, prefix_len), device=device, dtype=attention_mask.dtype), attention_mask],
                dim=1
            )
            torch.cuda.empty_cache()
            # Greedy, short, cache-enabled (fast & low-mem)
            gen_kwargs = {
                "inputs_embeds": gen_inputs_embeds,
                "attention_mask": gen_attn,
                "max_new_tokens": 1024,
                "min_new_tokens": 512,
                "early_stopping": True,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "num_beams": 2,
                "do_sample": True,
                "temperature": 0.25,
                "top_p": 0.9,
                "use_cache": True,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                # important when using DS/TP: avoids cross-GPU hangs
                "synced_gpus": False #True if torch.distributed.is_initialized() else False,
            }

            generated_tokens = model.llama.generate(**gen_kwargs)

            # teacher-forced loss (unchanged)
            loss = None
            if labels is not None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_embeds=prompt_embeds.to(device=device, dtype=target_dtype),
                    labels=labels.to(device),
                    ner_context_embeds=inputs.get("ner_context_embeds", None),
                    ner_gold_labels=inputs.get("ner_gold_labels", None),
                    entity_mask=inputs.get("entity_mask", None),
                    summary_ner_labels=inputs.get("summary_ner_labels", None),
                )
                loss = outputs["loss"].detach()

            return (loss, generated_tokens, labels)


# --------- Post-processing for generated notes ---------
SECTION_HEADERS_FULL = [
    "CHIEF COMPLAINT","HISTORY OF PRESENT ILLNESS","REVIEW OF SYSTEMS","VITALS",
    "PHYSICAL EXAMINATION","RESULTS","ASSESSMENT","PLAN","ASSESSMENT AND PLAN","INSTRUCTIONS"
]
def normalize_section_headers(note_body):
    
    for header in SECTION_HEADERS_FULL:
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

# --------- UniEval + metrics ---------
sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')
from utils import convert_to_json
from metric.evaluator import get_evaluator

# --- global, once ---
try:
    UNIEVAL = get_evaluator('summarization')  # keep your import path
except Exception:
    UNIEVAL = None  # fall back gracefully if anything odd happens

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
sacrebleu = evaluate.load("sacrebleu")
bleurt = evaluate.load("bleurt", config_name="bleurt-20")

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

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_str = [postprocess_generated_note(p) for p in pred_str]
    label_str = [canonicalize_note(l.strip()) for l in label_str]

    # ----- UniEval: reuse the already-loaded evaluator -----
    try:
        if UNIEVAL is not None:
            data = convert_to_json(output_list=pred_str, src_list=val_dialogues, ref_list=label_str)
            eval_scores = UNIEVAL.evaluate(data, print_result=True)
        else:
            eval_scores = {}
    except Exception:
        eval_scores = {}

    # ----- ROUGE (fast) -----
    results = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)

    # ----- BERTScore on GPU (a lot faster) -----
    bs_device = "cuda" if torch.cuda.is_available() else "cpu"
    bertscore_results = bertscore.compute(
        predictions=pred_str,
        references=label_str,
        lang="en",
        device=bs_device,
        batch_size=1
    )

    # ----- others (keep as-is) -----
    meteor_result = meteor.compute(predictions=pred_str, references=label_str)
    sacrebleu_result = sacrebleu.compute(predictions=pred_str, references=[[r] for r in label_str])
    try:
        bleurt_result = bleurt.compute(predictions=pred_str, references=label_str)
        bleurt_mean = float(np.mean(bleurt_result["scores"]))
    except Exception:
        bleurt_mean = float("nan")

    metrics = {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results["rougeLsum"],
        "bertscore_f1": float(np.mean(bertscore_results["f1"])),
        "meteor": float(meteor_result["meteor"]),
        "bleu": float(sacrebleu_result["score"]),
        "bleurt": bleurt_mean,
    }

    # attach UniEval if available
    if isinstance(eval_scores, list) and eval_scores and isinstance(eval_scores[0], dict):
        avg_unieval = {f"unieval_{k}": float(np.mean([d[k] for d in eval_scores])) for k in eval_scores[0].keys()}
        metrics.update(avg_unieval)
    elif isinstance(eval_scores, dict):
        metrics.update({f"unieval_{k}": v for k, v in eval_scores.items()})

    return metrics


# --------- DeepSpeed config auto-patch ---------
def load_and_patch_ds_config(path, use_bf16: bool):
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("zero_optimization", {}).setdefault("stage", 2)
    cfg.setdefault("fp16", {}); cfg.setdefault("bf16", {})
    cfg["fp16"]["enabled"] = not use_bf16
    cfg["bf16"]["enabled"] = bool(use_bf16)
    # keep optimizer offload disabled; param offload needs ZeRO-3
    return cfg

class MemoryCleanCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Called when eval finishes (right before training resumes)
        import gc, torch
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # helps defragment freed blocks
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def on_prediction_end(self, args, state, control, **kwargs):
        # Also clean after prediction-only passes
        import gc, torch
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass



# class EvalCleanupCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         # clear any left-over caches from generation
#         try:
#             model = kwargs.get("model", None)
#             if model is not None and hasattr(model, "llama"):
#                 # For some transformer versions with static cache impl
#                 if hasattr(model.llama, "clear_kv_cache"):
#                     model.llama.clear_kv_cache()
#                 if hasattr(model.llama, "past_key_values"):
#                     model.llama.past_key_values = None
#         except Exception:
#             pass
#         torch.cuda.empty_cache()


# =================== MAIN ===================
if __name__ == "__main__":
    # === Pick one of the two 70B models ===
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # Prefer Flash-Attn 2 if installed; else SDPA
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"

    cfg = AutoConfig.from_pretrained(model_name)
    setattr(cfg, "attn_implementation", attn_impl)
    setattr(cfg, "_attn_implementation", attn_impl)
    try: cfg.use_cache = False
    except Exception: pass

    # QLoRA 4-bit
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    gpu_cap = os.getenv("GPU_MAX_MEM", "46GiB")   # tweak if needed
    cpu_cap = os.getenv("CPU_MAX_MEM", "120GiB")  # tweak if needed

    # IMPORTANT: GPU keys must be integers (0,1,...) not "cuda:0"
    max_mem = {i: gpu_cap for i in range(n_gpus)}
    max_mem["cpu"] = cpu_cap

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map="auto",
        max_memory=max_mem
    )
    # QLoRA prep (casts norms, sets input grads off, enables GC)
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    try: base_model.config.use_cache = False
    except Exception: pass

    # ---- Data ----
    train_csv = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-TrainingSet.csv"
    val_csv   = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"
    train_df = pd.read_csv(train_csv); val_df = pd.read_csv(val_csv)
    train_dialogues = train_df["dialogue"].astype(str).tolist()
    train_notes     = train_df["note"].astype(str).tolist()
    val_dialogues   = val_df["dialogue"].astype(str).tolist()
    val_notes       = val_df["note"].astype(str).tolist()

    # ---- Stage1/2/3 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_model_dir = "/home/raza/EMRV1/EMRV2/outputs/stage1_outputs/final_model"
    stage1_model = AutoModelForTokenClassification.from_pretrained(stage1_model_dir).to(device)
    stage1_tokenizer = AutoTokenizer.from_pretrained(stage1_model_dir)

    backbone_name = "emilyalsentzer/Bio_ClinicalBERT"
    ENC_H_SIZE = stage1_model.config.hidden_size          # 768
    entity_label_count = stage1_model.config.num_labels

    stage2_encoder = BiasedAttentionEncoder(
        backbone_name=backbone_name,
        hidden_size=ENC_H_SIZE,
        entity_label_count=entity_label_count,
        bias_strength=3.0
    ).to(device)
    stage3_proj = PromptProjection(hidden_size=ENC_H_SIZE, prompt_len=16, mode="attentive")

    cache_train = "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/train70b_data_list.pt"
    cache_val   = "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/val70b_data_list.pt"
    if os.path.exists(cache_train) and os.path.exists(cache_val):
        print("Loading preprocessed data...")
        train_data_list = torch.load(cache_train)
        val_data_list   = torch.load(cache_val)
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
        torch.save(train_data_list, cache_train)
        torch.save(val_data_list, cache_val)
        print("Saved train70b_data_list.pt and val70b_data_list.pt")

    # ---- LoRA adapters (attention only = lower activation mem)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj","k_proj","v_proj","o_proj"]  # leaner than adding MLP
        target_modules=["q_proj", "v_proj"] 
    )

    model = PromptedLlamaMultiTask(
        base_model=base_model,
        prompt_len=16,
        encoder_hidden_size=ENC_H_SIZE,
        ner_label_count=7,
        soft_prompt_len=16,
        lora_cfg=lora_cfg
    )
    print_trainable_parameters(model)

    # ---- DeepSpeed: patch bf16/fp16 to match compute dtype
    ds_cfg_path = "/home/raza/EMRV1/EMRV2/ds_config_zero2.json"
    deepspeed_cfg = load_and_patch_ds_config(ds_cfg_path, use_bf16=(compute_dtype == torch.bfloat16))

    # ---- Truncation limits (env-tunable)
    TRAIN_MAX_LEN = int(os.getenv("TRAIN_MAX_LEN", "1536"))
    EVAL_MAX_LEN  = int(os.getenv("EVAL_MAX_LEN",  "1024"))

    training_args = TrainingArguments(
        output_dir="/home/raza/EMRV1/EMRV2/outputs/model_training_outputs/llama70b_joint_summ_ckpt",
        deepspeed=deepspeed_cfg,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        metric_for_best_model="rouge1",
        load_best_model_at_end=True,
        logging_steps=50,
        learning_rate=5e-5,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = JointMultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=MultiTaskSummarizationDataset(train_data_list),
        eval_dataset=MultiTaskSummarizationDataset(val_data_list),
        data_collator=DataCollatorTruncate(train_max_len=TRAIN_MAX_LEN, eval_max_len=EVAL_MAX_LEN, for_eval=False),
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            StepwiseLRSchedulerCallback(patience=2, lr_list=lr_list, mode="min", monitor="eval_loss"),
            MemoryCleanCallback(),
            # EvalCleanupCallback(),
        ],
        compute_metrics=compute_metrics
    )

    # HF calls GC on wrapper; ensure cache disabled:
    try: trainer.model.llama.config.use_cache = False
    except Exception: pass

    trainer.train()
    # trainer.save_model("/home/raza/EMRV1/EMRV2/outputs/model_training_outputs/llama70b_joint_summ_ckpt/final")
    print("Stage 4 LLM training complete!")

    eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
    if eval_logs:
        print("\nAverages:")
        keys = ['eval_loss','eval_rouge1','eval_rouge2','eval_rougeL','eval_rougeLsum','eval_bertscore_f1','eval_meteor','eval_bleu','eval_bleurt']
        for k in keys:
            vs = [log[k] for log in eval_logs if k in log]
            if vs: print(f"{k}: {np.mean(vs):.4f}")
        print("\nPer-epoch:")
        for i, log in enumerate(eval_logs):
            row = ", ".join([f"{k}: {log[k]:.4f}" for k in keys if k in log])
            print(f"Epoch {log.get('epoch', i+1)}: {row}")
    else:
        print("No evaluation logs found.")


















# # Stage4.py
# # Full fixed training script for DeepSeek-R1-Distill-Llama-70B with DeepSpeed + LoRA + multi-task logic.

# import os
# # reduce fragmentation
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# # keep TF off GPUs (for BLEURT)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# try:
#     import tensorflow as tf
#     try:
#         tf.config.set_visible_devices([], 'GPU')
#     except Exception:
#         pass
#     try:
#         tf.config.threading.set_intra_op_parallelism_threads(1)
#         tf.config.threading.set_inter_op_parallelism_threads(1)
#     except Exception:
#         pass
# except Exception:
#     pass

# import sys
# import json
# import re
# import subprocess
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     AutoModelForTokenClassification,
#     TrainerCallback,
#     EarlyStoppingCallback,
# )
# from peft import LoraConfig, get_peft_model, PeftModel
# import evaluate
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# # accelerate functions for safe sharded loading
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# # local modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from Stage2 import BiasedAttentionEncoder
# from Stage3 import PromptProjection
# from DataPreprocessing import preprocess_pipeline

# # Stage1 NER model path check
# ner_model_dir = "/home/raza/EMRV1/EMRV2/outputs/stage1_outputs/final_model/model.safetensors"
# if not os.path.exists(ner_model_dir):
#     print("NER model not found. Running Stage1.py...")
#     subprocess.run(["python", "/home/raza/EMRV1/EMRV2/Stage1.py"], check=True)
# else:
#     print("NER model found. Skipping Stage1 training.")

# # ---------------- text utils ----------------
# SECTION_HEADERS = [
#     "CHIEF COMPLAINT",
#     "HISTORY OF PRESENT ILLNESS",
#     "REVIEW OF SYSTEMS",
#     "VITALS",
#     "PHYSICAL EXAMINATION",
#     "RESULTS",
#     "ASSESSMENT",
#     "PLAN",
#     "ASSESSMENT AND PLAN",
#     "INSTRUCTIONS"
# ]

# def extract_sections(text, section_headers=SECTION_HEADERS):
#     sections = {}
#     pattern = '|'.join([re.escape(h) for h in section_headers])
#     parts = re.split(f"({pattern})", text)
#     curr_section = None
#     for part in parts:
#         part = part.strip()
#         if part in section_headers:
#             curr_section = part
#             sections[curr_section] = ""
#         elif curr_section and part:
#             if sections[curr_section]:
#                 sections[curr_section] += "\n" + part
#             else:
#                 sections[curr_section] = part
#     for h in section_headers:
#         if h not in sections:
#             sections[h] = ""
#     return sections

# def canonicalize_note(text, section_headers=SECTION_HEADERS):
#     sections = extract_sections(text, section_headers)
#     canonical = ""
#     for h in section_headers:
#         canonical += h + "\n\n" + sections[h].strip() + "\n\n"
#     return canonical.strip()

# def normalize_section_headers(note_body):
#     for header in SECTION_HEADERS:
#         pattern = rf"(?im)^{header.replace(' ', '[ _]')}[ \t]*:?"
#         note_body = re.sub(pattern, header, note_body)
#     return note_body

# def postprocess_generated_note(generated_text):
#     note_match = re.search(r"\[NOTE\]", generated_text, re.IGNORECASE)
#     if not note_match:
#         return generated_text.strip()
#     note_start = note_match.end()
#     after_note = generated_text[note_start:]
#     end_match = re.search(r"\[END OF NOTE\]|\[END NOTE\]|\[END INSTRUCTION\]|\[INSTRUCTION\]", after_note, re.IGNORECASE)
#     if end_match:
#         note_body = after_note[:end_match.start()]
#     else:
#         note_body = after_note
#     note_body = normalize_section_headers(note_body)
#     note_body = re.sub(r'\n{2,}', '\n\n', note_body)
#     return note_body.strip("\n\r\t ")

# # ---------------- presence helper ----------------
# def compute_entity_presence(entity_labels, num_entity_types):
#     mask = (entity_labels != -100)
#     masked = torch.where(mask, entity_labels, torch.zeros_like(entity_labels))
#     one_hot = F.one_hot(masked, num_classes=num_entity_types)
#     one_hot = one_hot * mask.unsqueeze(-1)
#     presence = (one_hot.sum(dim=1) > 0).float()
#     return presence

# # print LoRA trainable params
# def print_trainable_parameters(model):
#     print("\n==== Trainable parameters (should be LoRA adapters only!) ====")
#     total_params = 0
#     trainable_params = 0
#     for n, p in model.named_parameters():
#         total_params += p.numel()
#         if p.requires_grad:
#             trainable_params += p.numel()
#             print(f"Trainable: {n} {p.shape}")
#     print(f"Total parameters in model: {total_params:,}")
#     print(f"Total *trainable* parameters: {trainable_params:,}")
#     print("Trainable ratio: {:.4f}%".format(100 * trainable_params / total_params))
#     has_lora = False
#     if isinstance(model, PeftModel) or (hasattr(model, "llama") and isinstance(model.llama, PeftModel)):
#         print("LoRA is ACTIVE! Your model is correctly LoRA-wrapped.\n")
#         has_lora = True
#     else:
#         print("WARNING: LoRA is NOT active! Check model wrapping.\n")
#     return has_lora

# # ---------------- Custom wrapper with safe .to() ----------------
# class PromptedLlamaMultiTask(nn.Module):
#     def __init__(self, base_model, prompt_len, hidden_size, ner_label_count, soft_prompt_len=8, lora_cfg=None, consist_loss_fn=None):
#         super().__init__()
#         self.llama = base_model
#         self.prompt_len = prompt_len
#         self.hidden_size = hidden_size
#         self.llama_hidden_size = base_model.config.hidden_size
#         self.ner_label_count = ner_label_count
#         self.soft_prompt_len = soft_prompt_len

#         # small trainable components
#         self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, hidden_size).normal_(0, 0.02))
#         self.prompt_proj_to_llama = nn.Linear(self.hidden_size, self.llama_hidden_size)
#         self.ner_head = nn.Linear(hidden_size, ner_label_count)

#         # Apply LoRA to the underlying base model if requested
#         if lora_cfg is not None:
#             self.llama = get_peft_model(self.llama, lora_cfg)

#         self.consist_loss_fn = consist_loss_fn or self.default_consist_loss

#     def forward(self, input_ids, attention_mask, prompt_embeds, soft_prompt=None, labels=None, ner_context_embeds=None, ner_gold_labels=None, entity_mask=None, summary_ner_labels=None):
#         B, L = input_ids.shape
#         P = prompt_embeds.shape[1]
#         S = self.soft_prompt_len

#         target_dtype = self.prompt_proj_to_llama.weight.dtype
#         prompt_embeds = prompt_embeds.to(target_dtype)
#         if soft_prompt is None:
#             soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)
#         soft_prompt = soft_prompt.to(target_dtype)
#         prompt_embeds_proj = self.prompt_proj_to_llama(prompt_embeds)
#         soft_prompt_proj = self.prompt_proj_to_llama(soft_prompt)
#         input_embeds = self.llama.get_input_embeddings()(input_ids)
#         decoder_embeds = torch.cat([prompt_embeds_proj, soft_prompt_proj, input_embeds], dim=1)
#         full_mask = torch.cat([
#             torch.ones((B, P+S), dtype=attention_mask.dtype, device=attention_mask.device),
#             attention_mask
#         ], dim=1)
#         labels_padded = labels
#         if labels is not None:
#             pad_len = decoder_embeds.shape[1] - labels.shape[1]
#             if pad_len > 0:
#                 padding = labels.new_full((labels.shape[0], pad_len), -100)
#                 labels_padded = torch.cat([padding, labels], dim=1)
#             else:
#                 labels_padded = labels

#         llama_out = self.llama(inputs_embeds=decoder_embeds, attention_mask=full_mask, labels=labels_padded)
#         loss = getattr(llama_out, "loss", None)
#         if loss is None and isinstance(llama_out, dict):
#             loss = llama_out.get("loss", None)

#         ner_loss = None
#         if ner_context_embeds is not None and ner_gold_labels is not None:
#             target_dtype = self.ner_head.weight.dtype
#             ner_logits = self.ner_head(ner_context_embeds.to(target_dtype))
#             ner_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#             ner_loss = ner_loss_fct(ner_logits.view(-1, self.ner_label_count), ner_gold_labels.view(-1))

#         consist_loss = None
#         if (ner_gold_labels is not None) and (summary_ner_labels is not None):
#             consist_loss = self.consist_loss_fn(labels, summary_ner_labels, ner_gold_labels, entity_mask)

#         lmbd_ner = 0.2
#         lmbd_consist = 0.2
#         total_loss = loss
#         if ner_loss is not None:
#             total_loss = total_loss + lmbd_ner * ner_loss
#         if consist_loss is not None:
#             total_loss = total_loss + lmbd_consist * consist_loss

#         return {"loss": total_loss, "main_loss": loss, "ner_loss": ner_loss, "consist_loss": consist_loss}

#     def default_consist_loss(self, labels, summary_ner_labels, ner_gold_labels, entity_mask):
#         dialogue_entities = set(ner_gold_labels[ner_gold_labels != -100].cpu().numpy().tolist())
#         mask = (summary_ner_labels != -100)
#         missing = (mask) & (~torch.isin(summary_ner_labels, torch.tensor(list(dialogue_entities), device=summary_ner_labels.device)))
#         consist_loss = missing.float().mean()
#         return consist_loss

#     def to(self, *args, **kwargs):
#         # only move small modules and soft_prompt; do NOT move self.llama (the big model)
#         device = None
#         dtype = None
#         if len(args) > 0:
#             a0 = args[0]
#             if isinstance(a0, torch.dtype):
#                 dtype = a0
#             else:
#                 device = a0
#         if "device" in kwargs:
#             device = kwargs["device"]
#         if "dtype" in kwargs:
#             dtype = kwargs["dtype"]

#         if dtype is None:
#             if device is not None:
#                 self.prompt_proj_to_llama = self.prompt_proj_to_llama.to(device=device)
#                 self.ner_head = self.ner_head.to(device=device)
#                 self.soft_prompt.data = self.soft_prompt.data.to(device=device)
#         else:
#             self.prompt_proj_to_llama = self.prompt_proj_to_llama.to(device=device, dtype=dtype)
#             self.ner_head = self.ner_head.to(device=device, dtype=dtype)
#             self.soft_prompt.data = self.soft_prompt.data.to(device=device, dtype=dtype)
#         return self

# class MultiTaskSummarizationDataset(torch.utils.data.Dataset):
#     def __init__(self, data_list):
#         self.data_list = data_list
#     def __len__(self):
#         return len(self.data_list)
#     def __getitem__(self, idx):
#         return self.data_list[idx]

# # LR callback
# lr_list = [1e-4, 5e-5, 2e-5, 1e-5]
# class StepwiseLRSchedulerCallback(TrainerCallback):
#     def __init__(self, patience, lr_list, monitor="eval_loss", mode="min"):
#         self.patience = patience
#         self.lr_list = lr_list
#         self.monitor = monitor
#         self.mode = mode
#         self.best = None
#         self.bad_epochs = 0
#         self.lr_index = 0
#     def on_epoch_end(self, args, state, control, **kwargs):
#         logs = state.log_history[-1] if len(state.log_history) > 0 else {}
#         current = logs.get(self.monitor, None)
#         if current is None:
#             return
#         if self.best is None or current < self.best:
#             self.best = current
#             self.bad_epochs = 0
#         else:
#             self.bad_epochs += 1
#         if self.bad_epochs >= self.patience:
#             self.bad_epochs = 0
#             self.lr_index += 1
#             if self.lr_index < len(self.lr_list):
#                 new_lr = self.lr_list[self.lr_index]
#                 print(f"\n[StepwiseLR] Reducing LR to {new_lr} after plateau.")
#                 for group in kwargs["optimizer"].param_groups:
#                     group["lr"] = new_lr
#             else:
#                 print("\n[StepwiseLR] LR schedule exhausted.")

# # Custom Trainer
# class JointMultiTaskTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         prompt_embeds = inputs.pop('prompt_embeds')
#         ner_context_embeds = inputs.get('ner_context_embeds', None)
#         ner_gold_labels = inputs.get('ner_gold_labels', None)
#         entity_mask = inputs.get('entity_mask', None)
#         summary_ner_labels = inputs.get('summary_ner_labels', None)
#         output = model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             prompt_embeds=prompt_embeds,
#             labels=inputs.get('labels', None),
#             ner_context_embeds=ner_context_embeds,
#             ner_gold_labels=ner_gold_labels,
#             entity_mask=entity_mask,
#             summary_ner_labels=summary_ner_labels
#         )
#         loss = output["loss"]
#         return (loss, output) if return_outputs else loss

#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
#         model.eval()
#         with torch.no_grad():
#             device = next(model.parameters()).device
#             input_ids = inputs['input_ids'].to(device)
#             attention_mask = inputs['attention_mask'].to(device)
#             labels = inputs.get('labels', None)
#             if labels is not None:
#                 labels_mask = (labels[0] == -100)
#                 prompt_token_count = torch.sum(labels_mask).item()
#             else:
#                 raise ValueError("labels must be provided for slicing prompt.")
#             prompt_input_ids = input_ids[:, :prompt_token_count]
#             prompt_attention_mask = attention_mask[:, :prompt_token_count]
#             gen_kwargs = {
#                 "input_ids": prompt_input_ids,
#                 "attention_mask": prompt_attention_mask,
#                 "max_new_tokens": 512,
#                 "min_new_tokens": 64,
#                 "early_stopping": True,
#                 "eos_token_id": tokenizer.eos_token_id,
#                 "pad_token_id": tokenizer.pad_token_id,
#                 "repetition_penalty": 0.5,
#                 "no_repeat_ngram_size": 3,
#                 "num_beams": 1,
#                 "do_sample": False
#             }
#             generated_tokens = model.llama.generate(**gen_kwargs)
#             prompt_embeds = inputs.get('prompt_embeds', None)
#             loss = None
#             if labels is not None and input_ids is not None and prompt_embeds is not None:
#                 outputs = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     prompt_embeds=prompt_embeds.to(device=device, dtype=model.prompt_proj_to_llama.weight.dtype),
#                     labels=labels.to(device),
#                     ner_context_embeds=inputs.get("ner_context_embeds", None),
#                     ner_gold_labels=inputs.get("ner_gold_labels", None),
#                     entity_mask=inputs.get("entity_mask", None),
#                     summary_ner_labels=inputs.get("summary_ner_labels", None),
#                 )
#                 loss = outputs["loss"].detach()
#             return (loss, generated_tokens, labels)

# # UniEval + metrics
# sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')
# from utils import convert_to_json
# from metric.evaluator import get_evaluator

# rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")
# meteor = evaluate.load("meteor")
# sacrebleu = evaluate.load("sacrebleu")
# bleurt = evaluate.load("bleurt", config_name="bleurt-20")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     if isinstance(predictions, tuple):
#         predictions = predictions[0]
#     if isinstance(labels, tuple):
#         labels = labels[0]
#     if predictions.ndim == 3:
#         predictions = np.argmax(predictions, axis=-1)
#     predictions = predictions.astype(np.int32)
#     labels = labels.astype(np.int32)
#     labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
#     predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
#     pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     pred_str = [postprocess_generated_note(p) for p in pred_str]
#     label_str = [canonicalize_note(l.strip()) for l in label_str]
#     src_list = val_dialogues
#     data = convert_to_json(output_list=pred_str, src_list=src_list, ref_list=label_str)
#     evaluator = get_evaluator('summarization')
#     eval_scores = evaluator.evaluate(data, print_result=True)
#     results = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
#     bertscore_results = bertscore.compute(predictions=pred_str, references=label_str, lang="en")
#     meteor_result = meteor.compute(predictions=pred_str, references=label_str)
#     sacrebleu_result = sacrebleu.compute(predictions=pred_str, references=[[r] for r in label_str])
#     try:
#         bleurt_result = bleurt.compute(predictions=pred_str, references=label_str)
#         bleurt_mean = float(np.mean(bleurt_result["scores"]))
#     except Exception:
#         bleurt_mean = float("nan")
#     metrics = {
#         "rouge1": results["rouge1"],
#         "rouge2": results["rouge2"],
#         "rougeL": results["rougeL"],
#         "rougeLsum": results["rougeLsum"],
#         "bertscore_f1": float(np.mean(bertscore_results["f1"])),
#         "meteor": float(meteor_result["meteor"]),
#         "bleu": float(sacrebleu_result["score"]),
#         "bleurt": bleurt_mean,
#     }
#     print("Eval scores type:", type(eval_scores))
#     print("Eval scores repr:", repr(eval_scores))
#     if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
#         avg_unieval = {}
#         keys = eval_scores[0].keys()
#         for k in keys:
#             avg_unieval["unieval_" + k] = float(np.mean([d[k] for d in eval_scores]))
#         metrics.update(avg_unieval)
#     elif isinstance(eval_scores, dict):
#         metrics.update({f"unieval_{k}": v for k, v in eval_scores.items()})
#     else:
#         print("UniEval returned unknown type:", type(eval_scores))
#     return metrics

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     llama_model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"   #meta-llama/Meta-Llama-3-70B

#     # --- Read DeepSpeed config to determine precision ---
#     deepspeed_config = "/home/raza/EMRV1/EMRV2/ds_config_zero3_offload.json"
#     ds_fp16 = False
#     ds_bf16 = False
#     try:
#         with open(deepspeed_config, "r") as f:
#             ds_cfg = json.load(f)
#         if "fp16" in ds_cfg and isinstance(ds_cfg["fp16"], dict):
#             if ds_cfg["fp16"].get("enabled", False):
#                 ds_fp16 = True
#         if "bf16" in ds_cfg and isinstance(ds_cfg["bf16"], dict):
#             if ds_cfg["bf16"].get("enabled", False):
#                 ds_bf16 = True
#     except Exception as e:
#         print(f"Warning reading DS config: {e}")

#     if ds_fp16 and ds_bf16:
#         print("DeepSpeed config enables both fp16 and bf16 — defaulting to fp16.")
#         ds_bf16 = False

#     # hardware bf16 autodetect if DS didn't specify
#     bf16_hw = False
#     try:
#         if torch.cuda.is_available():
#             prop = torch.cuda.get_device_properties(0)
#             major = getattr(prop, "major", None)
#             if major is not None and major >= 8:
#                 bf16_hw = True
#     except Exception:
#         bf16_hw = False

#     if ds_fp16:
#         model_dtype = torch.float16
#         trainer_bf16 = False
#         trainer_fp16 = True
#         print("DeepSpeed requires FP16 -> using fp16.")
#     elif ds_bf16:
#         model_dtype = torch.bfloat16
#         trainer_bf16 = True
#         trainer_fp16 = False
#         print("DeepSpeed requires BF16 -> using bf16.")
#     else:
#         if bf16_hw:
#             model_dtype = torch.bfloat16
#             trainer_bf16 = True
#             trainer_fp16 = False
#             print("No DS mixed-precision set; bf16 supported -> using bf16.")
#         else:
#             model_dtype = torch.float16
#             trainer_bf16 = False
#             trainer_fp16 = True
#             print("No DS mixed-precision set; bf16 not detected -> using fp16.")

#     # ---------------- tokenizer ----------------
#     print(f"Loading tokenizer for {llama_model_path} ...")
#     tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
#     tokenizer.pad_token = tokenizer.eos_token

#     model_config = AutoConfig.from_pretrained(llama_model_path)

#     # Initialize empty model
#     with init_empty_weights():
#         base_model = AutoModelForCausalLM.from_config(model_config)

#     # Load checkpoint shards efficiently across GPUs
#     base_model = load_checkpoint_and_dispatch(
#         base_model,
#         llama_model_path,
#         device_map="auto",
#         no_split_module_classes=["LlamaDecoderLayer"],
#         dtype=torch.float16,
#         offload_folder="offload_dir",  # optional
#     )
#     print("Base model dispatched and loaded.")

#     # ---------------- Stage1/2/3 + data loading ----------------
#     train_csv = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-TrainingSet.csv"
#     val_csv = "/home/raza/EMRV1/EMRV2/datasets/ACI-TaskB-ValidationSet.csv"
#     train_df = pd.read_csv(train_csv)
#     val_df = pd.read_csv(val_csv)
#     train_dialogues = train_df["dialogue"].astype(str).tolist()
#     train_notes = train_df["note"].astype(str).tolist()
#     val_dialogues = val_df["dialogue"].astype(str).tolist()
#     val_notes = val_df["note"].astype(str).tolist()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     stage1_model_dir = "/home/raza/EMRV1/EMRV2/outputs/stage1_outputs/final_model"
#     stage1_model = AutoModelForTokenClassification.from_pretrained(stage1_model_dir).to(device)
#     stage1_tokenizer = AutoTokenizer.from_pretrained(stage1_model_dir)
#     backbone_name = "emilyalsentzer/Bio_ClinicalBERT"
#     hidden_size = stage1_model.config.hidden_size
#     entity_label_count = stage1_model.config.num_labels
#     stage2_encoder = BiasedAttentionEncoder(
#         backbone_name=backbone_name,
#         hidden_size=hidden_size,
#         entity_label_count=entity_label_count,
#         bias_strength=3.0
#     ).to(device)
#     stage3_proj = PromptProjection(
#         hidden_size=hidden_size,
#         prompt_len=16,
#         mode="attentive"
#     )

#     preprocess_train_cache = "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/train70b_data_list.pt"
#     preprocess_val_cache = "/home/raza/EMRV1/EMRV2/outputs/preprocessing_outputs/val70b_data_list.pt"
#     if os.path.exists(preprocess_train_cache) and os.path.exists(preprocess_val_cache):
#         print("Loading preprocessed data for 70B...")
#         train_data_list = torch.load(preprocess_train_cache)
#         val_data_list = torch.load(preprocess_val_cache)
#     else:
#         print("Preprocessing training set for 70B...")
#         train_data_list = preprocess_pipeline(train_dialogues, train_notes, stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj, tokenizer, device)
#         print("Preprocessing validation set for 70B...")
#         val_data_list = preprocess_pipeline(val_dialogues, val_notes, stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj, tokenizer, device)
#         torch.save(train_data_list, preprocess_train_cache)
#         torch.save(val_data_list, preprocess_val_cache)
#         print(f"Saved {preprocess_train_cache} and {preprocess_val_cache}")

#     train_dataset = MultiTaskSummarizationDataset(train_data_list)
#     val_dataset = MultiTaskSummarizationDataset(val_data_list)

#     # ---------------- LoRA and wrapper ----------------
#     ner_label_count = 7
#     prompt_len = 16
#     soft_prompt_len = 16
#     lora_cfg = LoraConfig(
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=["q_proj", "v_proj"]
#     )

#     model = PromptedLlamaMultiTask(
#         base_model=base_model,
#         prompt_len=prompt_len,
#         hidden_size=hidden_size,
#         ner_label_count=ner_label_count,
#         soft_prompt_len=soft_prompt_len,
#         lora_cfg=lora_cfg
#     )

#     # Attach config and HF-like attrs for DeepSpeed/HF checks
#     model.config = model.llama.config
#     model.device_map = getattr(model.llama, "device_map", None)
#     model.is_parallelizable = getattr(model.llama, "is_parallelizable", False)
#     model.model_parallel = getattr(model.llama, "model_parallel", False)

#     print_trainable_parameters(model)

#     # ---------------- TrainingArguments & Trainer ----------------
#     output_dir = "/home/raza/EMRV1/EMRV2/outputs/model_training_outputs/llama70b_deepseek_joint_summ_ckpt"
#     args = TrainingArguments(
#         output_dir=output_dir,
#         deepspeed=deepspeed_config,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=2,
#         num_train_epochs=3,
#         save_total_limit=1,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         warmup_ratio=0.1,
#         lr_scheduler_type="cosine",
#         metric_for_best_model="rouge1",
#         load_best_model_at_end=True,
#         logging_steps=10,
#         learning_rate=5e-5,
#         bf16=trainer_bf16,
#         fp16=trainer_fp16,
#         report_to="none",
#         dataloader_pin_memory=True,
#         remove_unused_columns=False,
#         ddp_find_unused_parameters=False,
#     )

#     trainer = JointMultiTaskTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         callbacks=[
#             EarlyStoppingCallback(early_stopping_patience=3),
#             StepwiseLRSchedulerCallback(patience=2, lr_list=lr_list, mode="min", monitor="eval_loss")
#         ],
#         compute_metrics=compute_metrics
#     )

#     # Start training
#     trainer.train()
#     trainer.save_model(os.path.join(output_dir, "final"))
#     print("Stage 4 LLM training complete!")

#     # Summarize eval logs
#     eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
#     if eval_logs:
#         print("\nAverage Evaluation Metrics (70B DeepSeek):")
#         metrics_keys = ['eval_loss', 'eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'eval_rougeLsum', 'eval_bertscore_f1', 'eval_meteor', 'eval_bleu', 'eval_bleurt']
#         for key in metrics_keys:
#             vals = [log[key] for log in eval_logs if key in log]
#             if vals:
#                 avg = np.mean(vals)
#                 print(f"{key}: {avg:.4f}")
#         print("\nPer-Epoch Evaluation Metrics:")
#         for i, log in enumerate(eval_logs):
#             epoch = log.get('epoch', i+1)
#             metrics_str = ", ".join([f"{k}: {log[k]:.4f}" for k in metrics_keys if k in log])
#             print(f"Epoch {epoch}: {metrics_str}")
#     else:
#         print("No evaluation logs found.")










# # file: stage4_prompted_lm_train.py
# file: Stage4_SAMSum.py   CUDA_VISIBLE_DEVICES=1,2,3,4 deepspeed --num_gpus=4 Stage4.py
# Usage: python Stage4_SAMSum.py
# Edit paths & LM_MODEL_PATH to your available LLM before running.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
# import os, sys, re, subprocess
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback, AutoModelForTokenClassification
# )
# from peft import LoraConfig, get_peft_model, PeftModel
# from datasets import load_dataset
# import evaluate
# from tqdm import tqdm

# # Make sure project root is on path so Stage1/Stage2/Stage3 and DataPreprocessing import
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# # ----------------- IMPORT YOUR UPDATED STAGES -----------------
# from Stage1 import NERDataset                # confirmed by you
# from Stage2 import BiasedAttentionEncoder    # confirmed by you
# from Stage3 import PromptProjection          # confirmed by you
# from DataPreprocessing import preprocess_pipeline

# # ---------------- CONFIG ----------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Stage-1 artifacts (trained for SAMSum)
# STAGE1_DIR = "/home/raza/EMRV1/EMRV2/outputs/stage111_outputs/final_model"   # set to your SAMSum stage1 output
# OUTDIR = "/home/raza/EMRV1/EMRV2/outputs/stage4_samsum"
# os.makedirs(OUTDIR, exist_ok=True)

# # LLM (set to your accessible LLM - LLaMA3-8B or a smaller model for a test run)
# LM_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"   # change if not available locally
# # If you don't have LLaMA-3-8B locally for quick testing, set LM_MODEL_PATH to "gpt2" temporarily.

# # LoRA config
# LORA_TARGET_MODULES = ["q_proj", "v_proj"]
# LORA_R = 8
# LORA_ALPHA = 32
# LORA_DROPOUT = 0.05

# # Training hyperparams (adjust to your cluster)
# EPOCHS = 3
# PER_DEVICE_BATCH_SIZE = 1
# GRAD_ACCUM = 2
# LR = 5e-5
# MAX_INPUT_LEN = 2048
# MAX_TARGET_LEN = 1024
# PROMPT_LEN = 16          # P
# SOFT_PROMPT_LEN = 16     # learned soft prompts
# BIAS_STRENGTH = 3.0
# backbone_name = "bert-base-cased"

# # SAMSum via HF
# HF_DATASET_ID = "knkarthick/samsum"   # uses the canonical samsum dataset
# # --------------------------------------------------------------

# # ---------------- Helpers / metrics ----------------
# import sys
# sys.path.append('/home/raza/EMRV1/EMRV2/UniEval')
# from utils import convert_to_json
# from metric.evaluator import get_evaluator

# rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")
# meteor = evaluate.load("meteor")
# sacrebleu = evaluate.load("sacrebleu")
# try:
#     bleurt = evaluate.load("bleurt", config_name="bleurt-20")
# except Exception:
#     bleurt = None

# def postprocess_generated_summary(text: str):
#     # Minimal cleanup for chat summarization outputs
#     return text.strip()

# def build_compute_metrics(tokenizer, src_list=None):
#     def compute_metrics(eval_pred):
#         preds, labels = eval_pred
#         if isinstance(preds, tuple): preds = preds[0]
#         preds = np.array(preds)
#         labels = np.array(labels)

#         # --- Decode ---
#         if preds.ndim == 3:
#             preds = np.argmax(preds, axis=-1)
#         labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
#         preds = np.where(preds == -100, tokenizer.pad_token_id, preds)

#         pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         pred_str = [p.strip() for p in pred_str]
#         label_str = [l.strip() for l in label_str]

#         # --- Metrics ---
#         res = {}
#         rouge_res = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
#         res.update({
#             "rouge1": rouge_res["rouge1"],
#             "rouge2": rouge_res["rouge2"],
#             "rougeL": rouge_res["rougeL"],
#             "rougeLsum": rouge_res.get("rougeLsum", rouge_res["rougeL"]),
#         })

#         br = bertscore.compute(predictions=pred_str, references=label_str, lang="en")
#         res["bertscore_f1"] = float(np.mean(br["f1"]))

#         try:
#             meteor_res = meteor.compute(predictions=pred_str, references=label_str)
#             res["meteor"] = float(meteor_res["meteor"])
#         except Exception:
#             res["meteor"] = float("nan")

#         try:
#             sb = sacrebleu.compute(predictions=pred_str, references=[[r] for r in label_str])
#             res["bleu"] = float(sb["score"])
#         except Exception:
#             res["bleu"] = float("nan")

#         if bleurt is not None:
#             try:
#                 brt = bleurt.compute(predictions=pred_str, references=label_str)
#                 res["bleurt"] = float(np.mean(brt["scores"]))
#             except Exception:
#                 res["bleurt"] = float("nan")
#         else:
#             res["bleurt"] = float("nan")
        
#         # --- UniEval Integration ---
#        # --------------- UniEval Integration ---------------
#         eval_scores = {}
#         if torch.distributed.is_initialized():
#             rank = torch.distributed.get_rank()
#         else:
#             rank = 0

#         if rank == 0:
#             try:
#                 # Ensure src_list available and has same length as preds
#                 if src_list is None:
#                     src_used = [""] * len(pred_str)
#                 else:
#                     src_used = list(src_list)
#                     if len(src_used) != len(pred_str):
#                         src_used = (src_used + [""] * len(pred_str))[:len(pred_str)]

#                 data_json = convert_to_json(
#                     output_list=pred_str,
#                     src_list=src_used,
#                     ref_list=label_str
#                 )

#                 evaluator = get_evaluator('summarization')
#                 eval_scores = evaluator.evaluate(data_json, print_result=False)

#             except Exception as e:
#                 print(f"[Rank 0] UniEval failed: {e}")
#                 eval_scores = {}

#         # Synchronize scores across ranks
#         if torch.distributed.is_initialized():
#             obj_list = [eval_scores]
#             torch.distributed.broadcast_object_list(obj_list, src=0)
#             eval_scores = obj_list[0]

#         # Parse UniEval results
#         if isinstance(eval_scores, list) and len(eval_scores) > 0 and isinstance(eval_scores[0], dict):
#             avg_unieval = {}
#             keys = eval_scores[0].keys()
#             for k in keys:
#                 avg_unieval[f"unieval_{k}"] = float(np.mean([d[k] for d in eval_scores]))
#             res.update(avg_unieval)
#         elif isinstance(eval_scores, dict):
#             res.update({f"unieval_{k}": float(v) for k, v in eval_scores.items()})

#         return res
#     return compute_metrics


# # ----------------- Model & dataset classes (adapted from your Stage4) -----------------

# class PromptedLlamaMultiTask(nn.Module):
#     """
#     Same structure as your Stage4 class but adapted to SAMSum pipeline calling signature.
#     """
#     def __init__(self, base_model, prompt_len, hidden_size, ner_label_count,
#                  soft_prompt_len=16, lora_cfg=None, consist_loss_fn=None):
#         super().__init__()
#         self.llama = base_model
#         # wrap LM with LoRA if cfg provided
#         if lora_cfg is not None:
#             self.llama = get_peft_model(self.llama, lora_cfg)
#         self.prompt_len = prompt_len
#         self.hidden_size = hidden_size
#         self.llama_hidden_size = base_model.config.hidden_size
#         self.ner_label_count = ner_label_count
#         self.soft_prompt_len = soft_prompt_len
#         self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, hidden_size).normal_(0, 0.02))
#         self.prompt_proj_to_llama = nn.Linear(self.hidden_size, self.llama_hidden_size)
#         self.ner_head = nn.Linear(hidden_size, ner_label_count)
#         self.consist_loss_fn = consist_loss_fn or self.default_consist_loss

#     def forward(self,
#                 input_ids, attention_mask, prompt_embeds,
#                 labels=None, ner_context_embeds=None, ner_gold_labels=None,
#                 entity_mask=None, summary_ner_labels=None):
#         B = input_ids.size(0)
#         P = prompt_embeds.size(1)
#         S = self.soft_prompt_len

#         dtype = self.prompt_proj_to_llama.weight.dtype
#         prompt_embeds = prompt_embeds.to(dtype)
#         soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(dtype)
#         prompt_embeds_proj = self.prompt_proj_to_llama(prompt_embeds)
#         soft_prompt_proj = self.prompt_proj_to_llama(soft_prompt)
#         input_embeds = self.llama.get_input_embeddings()(input_ids)

#         decoder_embeds = torch.cat([prompt_embeds_proj, soft_prompt_proj, input_embeds], dim=1)
#         full_mask = torch.cat([torch.ones((B, P + S), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=1)

#         labels_padded = labels
#         if labels is not None:
#             pad_len = decoder_embeds.shape[1] - labels.shape[1]
#             if pad_len > 0:
#                 padding = labels.new_full((labels.shape[0], pad_len), -100)
#                 labels_padded = torch.cat([padding, labels], dim=1)
#             else:
#                 labels_padded = labels

#         out = self.llama(inputs_embeds=decoder_embeds, attention_mask=full_mask, labels=labels_padded)
#         loss = out.loss
#         logits = out.logits
#         ner_loss = None
#         if ner_context_embeds is not None and ner_gold_labels is not None:
#             ner_logits = self.ner_head(ner_context_embeds.to(self.ner_head.weight.dtype))
#             ner_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#             ner_loss = ner_loss_fct(ner_logits.view(-1, self.ner_label_count), ner_gold_labels.view(-1))
#         else:
#             ner_logits = torch.zeros(1, device=decoder_embeds.device)
#         consist_loss = None
#         if (ner_gold_labels is not None) and (summary_ner_labels is not None):
#             consist_loss = self.consist_loss_fn(labels, summary_ner_labels, ner_gold_labels, entity_mask)
#         else:
#             consist_loss = torch.zeros(1, device=decoder_embeds.device)
#         lmbd_ner = 0.2
#         lmbd_consist = 0.2
#         total_loss = loss
#         if ner_loss is not None: total_loss = total_loss + lmbd_ner * ner_loss
#         if consist_loss is not None: total_loss = total_loss + lmbd_consist * consist_loss

#         return {"loss": total_loss, 
#                 "main_loss": loss, 
#                 "ner_loss": ner_loss if ner_loss is not None else torch.tensor(0.0, device=decoder_embeds.device), 
#                 "consist_loss": consist_loss if consist_loss is not None else torch.tensor(0.0, device=decoder_embeds.device),
#                 "logits": logits}

#     def default_consist_loss(self, labels, summary_ner_labels, ner_gold_labels, entity_mask):
#         # binary presence penalty (simple)
#         dialogue_entities = set(ner_gold_labels[ner_gold_labels != -100].cpu().tolist())
#         mask = (summary_ner_labels != -100)
#         missing = (mask) & (~torch.isin(summary_ner_labels, torch.tensor(list(dialogue_entities), device=summary_ner_labels.device)))
#         return missing.float().mean()

# class MultiTaskSummarizationDataset(torch.utils.data.Dataset):
#     def __init__(self, data_list):
#         # data_list is produced by preprocess_pipeline (list of dicts matching previous pipeline)
#         self.data_list = data_list
#     def __len__(self): return len(self.data_list)
#     def __getitem__(self, idx): return self.data_list[idx]

# # Trainer subclass (use your original JointMultiTaskTrainer logic)
# from transformers import Trainer as HfTrainer
# class JointMultiTaskTrainer(HfTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         prompt_embeds = inputs.pop('prompt_embeds')
#         ner_context_embeds = inputs.get('ner_context_embeds', None)
#         ner_gold_labels = inputs.get('ner_gold_labels', None)
#         entity_mask = inputs.get('entity_mask', None)
#         summary_ner_labels = inputs.get('summary_ner_labels', None)
#         output = model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             prompt_embeds=prompt_embeds,
#             labels=inputs['labels'],
#             ner_context_embeds=ner_context_embeds,
#             ner_gold_labels=ner_gold_labels,
#             entity_mask=entity_mask,
#             summary_ner_labels=summary_ner_labels
#         )
#         loss = output["loss"]
#         return (loss, output) if return_outputs else loss
#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
#         """
#         Perform generation conditioned on prompt prefix, then compute loss for logging.
#         Returns (loss, generated_tokens, labels)
#         """
#         model.eval()
#         with torch.no_grad():
#             device = next(model.parameters()).device

#             input_ids = inputs['input_ids'].to(device)
#             attention_mask = inputs['attention_mask'].to(device)
#             labels = inputs.get('labels', None)
#             if labels is not None:
#                 labels = labels.to(device)

#             # prompt_embeds is tensor [B, P, H_stage3] (precomputed by preprocess_pipeline)
#             prompt_embeds = inputs.get('prompt_embeds', None)
#             if prompt_embeds is not None:
#                 # keep prompt_embeds in original dtype; model will cast internally
#                 prompt_embeds = prompt_embeds.to(device)

#             # We assume labels have prefix positions marked as -100 (these correspond to prompt + soft tokens)
#             # Determine prompt token count using the first example in batch (works when prefix length is uniform)
#             if labels is None:
#                 # no labels -> generate only
#                 prompt_token_count = 0
#             else:
#                 labels_mask = (labels[0] == -100)
#                 prompt_token_count = int(torch.sum(labels_mask).item())

#             if prompt_token_count <= 0:
#                 # No explicit prompt token ids in input_ids; fall back to using full input_ids as condition
#                 gen_input_ids = input_ids
#                 gen_attention_mask = attention_mask
#             else:
#                 gen_input_ids = input_ids[:, :prompt_token_count]
#                 gen_attention_mask = attention_mask[:, :prompt_token_count]

#             # ---------- generation ----------
#             gen_kwargs = dict(
#                 max_new_tokens=MAX_TARGET_LEN,
#                 min_new_tokens=512,
#                 num_beams=2,
#                 early_stopping=True,
#                 # do_sample=True,
#                 top_p=0.9,
#                 temperature=0.25,    
#                 do_sample=True,
#                 use_cache=True,
#                 repetition_penalty=1.2,
#                 no_repeat_ngram_size=3,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )

#             # Call generate on underlying llama model; this uses token ids (prefix) to condition generation
#             # If your true conditioning requires prompt_embeds (not token prefix), you may need to adapt generation.
#             generated = model.llama.generate(
#                 input_ids=gen_input_ids,
#                 attention_mask=gen_attention_mask,
#                 **gen_kwargs
#             )

#             # ---------- compute loss (for logging) ----------
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 prompt_embeds=prompt_embeds,
#                 labels=labels,
#             )
#             loss = outputs["loss"].detach().float()

#             # Ensure generated is a tensor (not list). HuggingFace generate returns torch.LongTensor
#             if isinstance(generated, torch.Tensor):
#                 gen_tensor = generated
#             else:
#                 gen_tensor = torch.tensor(generated, device=device)

#             # Return tuple expected by Trainer: (loss, preds, labels)
#             return (loss, gen_tensor, labels)

# import os
# import torch

# PREPROCESS_DIR = os.path.join(OUTDIR, "preprocessing")
# os.makedirs(PREPROCESS_DIR, exist_ok=True)

# def preprocess_or_load(
#     dialogues, summaries,
#     stage1_model, stage1_tokenizer,
#     stage2_encoder, stage3_proj,
#     llama_tokenizer,
#     split_name="train",
#     device="cuda"
# ):
#     """
#     Check if preprocessed file exists. If yes, load from disk.
#     If not, run preprocess_pipeline and save the results.
#     """
#     cached_file = os.path.join(PREPROCESS_DIR, f"{split_name}_preprocessed.pt")
#     if os.path.exists(cached_file):
#         print(f"[INFO] Loading cached preprocessed {split_name} data from {cached_file}")
#         data_list = torch.load(cached_file)
#     else:
#         print(f"[INFO] Preprocessing {split_name} split...")
#         data_list = preprocess_pipeline(
#             dialogues, summaries,
#             stage1_model, stage1_tokenizer,
#             stage2_encoder, stage3_proj,
#             llama_tokenizer,
#             device=device
#         )
#         torch.save(data_list, cached_file)
#         print(f"[INFO] Saved preprocessed {split_name} data to {cached_file}")
#     return data_list
# # ---------------- Main ----------------
# import os
# import torch

# if torch.cuda.is_available():
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     DEVICE = torch.device("cuda", local_rank)
# else:
#     DEVICE = torch.device("cpu")

# def main():
#     print("Loading SAMSum dataset...")
#     ds = load_dataset(HF_DATASET_ID)
#     train_dialogues = ds["train"]["dialogue"]
#     train_summaries = ds["train"]["summary"]
#     val_dialogues = ds["validation"]["dialogue"]
#     val_summaries = ds["validation"]["summary"]

#     # Stage1 load (SAMSum-finetuned NER you built earlier)
#     print("Loading Stage1 NER pipeline...")
#     stage1 = AutoModelForTokenClassification.from_pretrained(STAGE1_DIR)
#     stage1_tokenizer = getattr(stage1, "tokenizer", None)
#     if stage1_tokenizer is None:
#         # fallback to loading tokenizer from STAGE1_DIR
#         from transformers import AutoTokenizer
#         stage1_tokenizer = AutoTokenizer.from_pretrained(STAGE1_DIR, use_fast=True)

#     # instantiate stage2/3 using SAMSum appropriate configs
#     hidden_size = stage1.config.hidden_size if hasattr(stage1, "config") else 768
#     ner_label_count = stage1.config.num_labels if hasattr(stage1, "config") else 10
#     print("Stage1 hidden_size:", hidden_size, "ner_label_count:", ner_label_count)

#     stage2 = BiasedAttentionEncoder(
#         base_model_name=backbone_name,
#         # hidden_size=hidden_size,
#         entity_label_count=ner_label_count,
#         bias_strength=BIAS_STRENGTH
#     )

#     stage3 = PromptProjection(
#         hidden_size=hidden_size,
#         prompt_count=PROMPT_LEN if 'PROMPT_LEN' in globals() else PROMPT_LEN,
#         projection_mode="attentive"
#     )

#     # LLM and tokenizer
#     print("Loading LLM:", LM_MODEL_PATH)
#     tokenizer_local = AutoTokenizer.from_pretrained(LM_MODEL_PATH, use_fast=True)
#     base_model = AutoModelForCausalLM.from_pretrained(LM_MODEL_PATH)
#     if tokenizer_local.pad_token is None:
#         tokenizer_local.pad_token = tokenizer_local.eos_token

#     # Preprocess dataset with your pipeline (returns lists ready for MultiTaskSummarizationDataset)
#     print("Running preprocess_pipeline on SAMSum (this may take a while)...")
#     train_data_list = preprocess_or_load(
#     list(train_dialogues), list(train_summaries),
#     stage1, stage1_tokenizer, stage2, stage3, tokenizer_local,
#     split_name="train", device=DEVICE
#     )

#     val_data_list = preprocess_or_load(
#         list(val_dialogues), list(val_summaries),
#         stage1, stage1_tokenizer, stage2, stage3, tokenizer_local,
#         split_name="val", device=DEVICE
#     )

#     print("Preprocessing complete. Examples:", len(train_data_list), len(val_data_list))

#     # wrap datasets
#     train_ds = MultiTaskSummarizationDataset(train_data_list)
#     val_ds = MultiTaskSummarizationDataset(val_data_list)

#     # Model wrapper with LoRA
#     lora_cfg = LoraConfig(
#         r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
#         bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES
#     )

#     torch.cuda.set_per_process_memory_fraction(0.9)
    
#     model = PromptedLlamaMultiTask(
#         base_model=base_model,
#         prompt_len=PROMPT_LEN,
#         hidden_size=hidden_size,
#         ner_label_count=ner_label_count,
#         soft_prompt_len=SOFT_PROMPT_LEN,
#         lora_cfg=lora_cfg
#     )
#     model.llama.gradient_checkpointing_enable()

    

#     model.to(DEVICE)
#     # Print LoRA trainable params
#     print("Model summary (trainable parameters):")
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total: {total_params:,}, Trainable: {trainable:,} ({100*trainable/total_params:.4f}%)")

#     # Trainer args
#     args = TrainingArguments(
#         output_dir=os.path.join(OUTDIR, "checkpoints"),
#         deepspeed="ds_config.json",
#         per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
#         per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
#         gradient_accumulation_steps=GRAD_ACCUM,
#         num_train_epochs=EPOCHS,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         save_total_limit=2,
#         learning_rate=LR,
#         fp16=True,
#         logging_steps=50,
#         report_to="none",
#         load_best_model_at_end=True,
#         dataloader_pin_memory=False,
#         metric_for_best_model="rouge1"
#     )

#     # data collator: preprocess_pipeline already returns dicts matching expected inputs, simple collate:
#     def collate_fn(batch):
#         # batch: list of dicts. Stack elements safely to tensors on device
#         keys = batch[0].keys()
#         out = {}
#         for k in keys:
#             vals = [b[k] for b in batch]
#             if isinstance(vals[0], torch.Tensor):
#                 out[k] = torch.stack(vals)
#             else:
#                 # assume prompt_embeds is numpy/tensor already - ensure tensor
#                 out[k] = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in vals])
#         return out

#     trainer = JointMultiTaskTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         data_collator=collate_fn,
#         tokenizer=tokenizer_local,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
#         compute_metrics=build_compute_metrics(tokenizer_local, src_list=val_dialogues)
#     )

#     print("Starting training on SAMSum...")
#     trainer.train()
#     print("Saving final model & adapters...")
#     trainer.save_model(os.path.join(OUTDIR, "final"))
#     # Save prompt adapter (projector weights + soft prompt)
#     torch.save({
#         "proj_state": model.prompt_proj.state_dict(),
#         "soft_prompt": model.soft_prompt.detach().cpu(),
#         "stage_hidden": hidden_size
#     }, os.path.join(OUTDIR, "prompt_adapter.pt"))

#     print("SAMSum Stage4 training complete. Outputs in:", OUTDIR)

# if __name__ == "__main__":
#     main()
