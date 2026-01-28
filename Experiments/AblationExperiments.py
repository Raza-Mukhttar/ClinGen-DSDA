# Ablation joint lllm training 

import torch
import pandas as pd
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model, PeftModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Stage2_BiasedAttentionEncoder import BiasedAttentionEncoder
from Stage3_PromptInjection import PromptProjection
from AblationPreprocessing import preprocess_pipeline  
import evaluate
import numpy as np

# =================== Consistency Loss Helper ===================
import torch.nn.functional as F

def compute_entity_presence(entity_labels, num_entity_types):
    mask = (entity_labels != -100)
    masked = torch.where(mask, entity_labels, torch.zeros_like(entity_labels))
    one_hot = F.one_hot(masked, num_classes=num_entity_types)
    one_hot = one_hot * mask.unsqueeze(-1)
    presence = (one_hot.sum(dim=1) > 0).float()
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

    has_lora = False
    if hasattr(model, "llama") and isinstance(model.llama, PeftModel):
        print("LoRA is ACTIVE! Your model is correctly LoRA-wrapped.\n")
        has_lora = True
    else:
        print("WARNING: LoRA is NOT active! Check model wrapping.\n")
    return has_lora

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
        prompt_embeds_proj = self.prompt_proj_to_llama(prompt_embeds)
        soft_prompt_proj = self.prompt_proj_to_llama(soft_prompt)
        input_embeds = self.llama.get_input_embeddings()(input_ids)

        decoder_embeds = torch.cat([prompt_embeds_proj, soft_prompt_proj, input_embeds], dim=1)
        full_mask = torch.cat([
            torch.ones((B, P+S), dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask
        ], dim=1)

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
        dialogue_entities = set(ner_gold_labels[ner_gold_labels != -100].cpu().numpy().tolist())
        mask = (summary_ner_labels != -100)
        missing = (mask) & (~torch.isin(summary_ner_labels, torch.tensor(list(dialogue_entities), device=summary_ner_labels.device)))
        consist_loss = missing.float().mean()
        return consist_loss

    def generate(self, *args, **kwargs):
        return self.llama.generate(*args, **kwargs)

class MultiTaskSummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

class JointMultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
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
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            prompt_embeds = inputs.pop('prompt_embeds').to(device).to(dtype=model.prompt_proj_to_llama.weight.dtype)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs.get("labels", None)
            if labels is not None:
                labels = labels.to(device)
            projected_prompt = model.prompt_proj_to_llama(prompt_embeds)
            input_embeds = model.llama.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([projected_prompt, input_embeds], dim=1)
            prefix_attention = torch.ones(projected_prompt.size()[:-1], dtype=attention_mask.dtype, device=device)
            full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": full_attention_mask,
                "max_new_tokens": 1024,
                "num_beams": 6,
                "do_sample": False,
                "pad_token_id": model.llama.config.eos_token_id,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }
            generated_tokens = model.generate(**gen_kwargs)
            if labels is not None and not prediction_loss_only:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    prompt_embeds=prompt_embeds,
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            else:
                loss = torch.tensor(0.0, device=device)
            return loss, generated_tokens, labels

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
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    pred_str = [p.strip() for p in pred_str]
    label_str = [l.strip() for l in label_str]
    print("Sample pred:", pred_str[0])
    print("Sample ref:", label_str[0])
    print("All pred lens:", [len(p) for p in pred_str])
    print("All ref lens:", [len(l) for l in label_str])
    results = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
    bertscore_results = bertscore.compute(predictions=pred_str, references=label_str, lang="en")
    metrics = {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results["rougeLsum"],
        "bertscore_f1": float(np.mean(bertscore_results["f1"])),
    }
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
    ner_label_count = 7  # as per Stage 1
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # ========== DATA LOADING ==========
    train_csv = "/datasets/ACI-TaskB-TrainingSet.csv"
    val_csv = "/datasets/ACI-TaskB-ValidationSet.csv"
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_dialogues = train_df["dialogue"].astype(str).tolist()
    train_notes = train_df["note"].astype(str).tolist()
    val_dialogues = val_df["dialogue"].astype(str).tolist()
    val_notes = val_df["note"].astype(str).tolist()

    # ========== STAGE 1, 2, 3 MODEL LOADING ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_model_dir = "outputs/stage1_outputs/final_model"
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

    # ========== PREPROCESS: Choose Your Ablation Here ==========
    # Ablation options: skip_ner, skip_bias, skip_prompt, skip_summary_ner
    # ablation_mode = None
    # ablation_mode = {"skip_ner": True}         # No Stage 1 NER
    ablation_mode = {"skip_bias": True}        # No Stage 2 Biased Attention
    # ablation_mode = {"skip_prompt": True}      # No Prompt Injection
    # ablation_mode = {"skip_summary_ner": True} # No summary NER for consistency
    # ablation_mode = {"skip_ner": True, "skip_bias": True}  # Combo ablation

    train_data_list = preprocess_pipeline(
        train_dialogues, train_notes,
        stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
        tokenizer, device,
        ablation=ablation_mode
    )
    val_data_list = preprocess_pipeline(
        val_dialogues, val_notes,
        stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj,
        tokenizer, device,
        ablation=ablation_mode
    )

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
        output_dir="outputs/llama_joint_summ_ckpt/run4",
        deepspeed="./ds_config_zero2.json",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
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
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("outputs/llama_joint_summ_ckpt/run4/final")
    print("Stage 4 LLM training complete!")

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
        print("\nPer-Epoch Evaluation Metrics:")
        for i, log in enumerate(eval_logs):
            print(f"Epoch {log.get('epoch', i+1)}: " + ", ".join([f"{k}: {log[k]:.4f}" for k in metrics_keys if k in log]))
    else:
        print("No evaluation logs found.")