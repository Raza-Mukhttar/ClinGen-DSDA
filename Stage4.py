# =============================================================================
#
# Recommended run:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 python Train_MultiSeed_Llama70B_ORIGINAL_QLORA_MULTI_DEVICE_FIXED.py --seeds 42
#
# =============================================================================

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

# Keep TensorFlow/BLEURT off GPUs.
try:
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
except Exception:
    pass

import argparse
import gc
import json
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# =============================================================================
# 
# =============================================================================

PROJECT_ROOT = os.environ.get("CLINGEN_ROOT", "/home/project")
os.chdir(PROJECT_ROOT) if os.path.isdir(PROJECT_ROOT) else None
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Stage2 import BiasedAttentionEncoder
    from Stage3 import PromptProjection
except Exception:
    from Stage2_BiasedAttentionEncoder import BiasedAttentionEncoder
    from Stage3_PromptInjection import PromptProjection

try:
    from DataPreprocessing import preprocess_pipeline
except Exception:
    from Data_Preprocessing import preprocess_pipeline

UNIEVAL_DIR = os.path.join(PROJECT_ROOT, "UniEval")
if os.path.isdir(UNIEVAL_DIR):
    sys.path.insert(0, UNIEVAL_DIR)
try:
    from utils import convert_to_json
    from metric.evaluator import get_evaluator
except Exception:
    convert_to_json = None
    get_evaluator = None


MODEL_NAME = os.environ.get("LLAMA70B_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "ACI-TaskB-TrainingSet.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "datasets", "ACI-TaskB-ValidationSet.csv")
STAGE1_MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "stage1_outputs", "final_model")
STAGE1_MODEL_FILE = os.path.join(STAGE1_MODEL_DIR, "model.safetensors")
STAGE1_SCRIPT = os.environ.get("STAGE1_SCRIPT", os.path.join(PROJECT_ROOT, "Stage1_NER_Module.py"))
if not os.path.exists(STAGE1_SCRIPT):
    _stage1_fallback = os.path.join(PROJECT_ROOT, "Stage1.py")
    if os.path.exists(_stage1_fallback):
        STAGE1_SCRIPT = _stage1_fallback

CACHE_TRAIN = os.path.join(PROJECT_ROOT, "outputs", "preprocessing_outputs_70b", "train_data_list.pt")
CACHE_VAL = os.path.join(PROJECT_ROOT, "outputs", "preprocessing_outputs_70b", "val_data_list.pt")
MULTISEED_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "multiseed", "llama70b_original_qlora")

# =============================================================================
# HYPERPARAMETERS — best stable setting for original QLoRA architecture
# =============================================================================

SEEDS = [42, 123, 7]
PROMPT_LEN = 16
SOFT_PROMPT_LEN = 16
NER_LABEL_COUNT = 5
BACKBONE_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Keep original attention-only LoRA architecture but slightly stronger than q/v only.
# q/v only is fastest; q/k/v/o improves note structure without the instability of MLP LoRA on tiny data.
LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = os.environ.get("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")

LEARNING_RATE = float(os.environ.get("LR", "2e-5"))
NUM_EPOCHS = int(os.environ.get("EPOCHS", "5"))
TRAIN_BS = int(os.environ.get("TRAIN_BS", "1"))
EVAL_BS = int(os.environ.get("EVAL_BS", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "2"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "0.5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))

TRAIN_MAX_LEN = int(os.environ.get("TRAIN_MAX_LEN", "2048"))
EVAL_MAX_LEN = int(os.environ.get("EVAL_MAX_LEN", "1536"))

GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "512"))
GEN_MIN_NEW_TOKENS = int(os.environ.get("GEN_MIN_NEW_TOKENS", "128"))
GEN_NUM_BEAMS = int(os.environ.get("GEN_NUM_BEAMS", "4"))
GEN_DO_SAMPLE = os.environ.get("GEN_DO_SAMPLE", "false").lower() == "true"
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "0.9"))
GEN_REPETITION_PENALTY = float(os.environ.get("GEN_REPETITION_PENALTY", "1.10"))
GEN_NO_REPEAT_NGRAM = int(os.environ.get("GEN_NO_REPEAT_NGRAM", "3"))

# Heavy metrics: kept because original script evaluates them, but loaded lazily once.
ENABLE_UNIEVAL = os.environ.get("ENABLE_UNIEVAL", "1") == "1"
ENABLE_BERTSCORE = os.environ.get("ENABLE_BERTSCORE", "1") == "1"
ENABLE_BLEURT = os.environ.get("ENABLE_BLEURT", "1") == "1"

SECTION_HEADERS = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS", "VITALS", "RESULTS", "INSTRUCTIONS"]
SECTION_HEADERS_FULL = [
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS", "VITALS",
    "PHYSICAL EXAMINATION", "RESULTS", "ASSESSMENT", "PLAN", "ASSESSMENT AND PLAN", "INSTRUCTIONS"
]

# =============================================================================
# UTILS
# =============================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    print(f"[Seed] Set all seeds to {seed}")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def extract_sections(text: str, section_headers=SECTION_HEADERS) -> Dict[str, str]:
    sections = {}
    pattern = "|".join([re.escape(h) for h in section_headers])
    parts = re.split(f"({pattern})", str(text))
    curr_section = None
    for part in parts:
        part = part.strip()
        if part in section_headers:
            curr_section = part
            sections[curr_section] = ""
        elif curr_section and part:
            sections[curr_section] = (sections[curr_section] + "\n" + part) if sections[curr_section] else part
    for h in section_headers:
        sections.setdefault(h, "")
    return sections


def canonicalize_note(text: str, section_headers=SECTION_HEADERS) -> str:
    sections = extract_sections(text, section_headers)
    canonical = ""
    for h in section_headers:
        canonical += h + "\n\n" + sections[h].strip() + "\n\n"
    return canonical.strip()


def normalize_section_headers(note_body: str) -> str:
    for header in SECTION_HEADERS_FULL:
        pattern = rf"(?im)^{header.replace(' ', '[ _]')}[ \t]*:?"
        note_body = re.sub(pattern, header, note_body)
    return note_body


def postprocess_generated_note(generated_text: str) -> str:
    generated_text = str(generated_text)
    # Remove repeated prompt echo if present.
    note_match = re.search(r"\[NOTE\]", generated_text, re.IGNORECASE)
    if note_match:
        after_note = generated_text[note_match.end():]
    else:
        # If [NOTE] is absent, try to cut away instruction block.
        m = re.search(r"(?:CHIEF COMPLAINT|HISTORY OF PRESENT ILLNESS|REVIEW OF SYSTEMS|VITALS|RESULTS|INSTRUCTIONS)", generated_text, re.IGNORECASE)
        after_note = generated_text[m.start():] if m else generated_text
    end_match = re.search(r"\[END OF NOTE\]|\[END NOTE\]|\[END INSTRUCTION\]|\[INSTRUCTION\]", after_note, re.IGNORECASE)
    note_body = after_note[:end_match.start()] if end_match else after_note
    note_body = normalize_section_headers(note_body)
    note_body = re.sub(r"\n{3,}", "\n\n", note_body)
    return note_body.strip("\n\r\t ")


def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters visible: {total:,}")
    print(f"Total trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {100.0 * trainable / max(total, 1):.4f}%")
    if hasattr(model, "llama") and isinstance(model.llama, PeftModel):
        print("LoRA is ACTIVE.\n")
        try:
            model.llama.print_trainable_parameters()
        except Exception:
            pass
    else:
        print("WARNING: LoRA is NOT active.\n")

# =============================================================================
# 
# =============================================================================

class PromptedLlamaMultiTask(nn.Module):
    def __init__(self, base_model, prompt_len, encoder_hidden_size, ner_label_count,
                 soft_prompt_len=8, lora_cfg=None, consist_loss_fn=None):
        super().__init__()
        self.llama = base_model
        self.prompt_len = prompt_len
        self.encoder_hidden_size = encoder_hidden_size
        self.llama_hidden_size = base_model.config.hidden_size
        self.ner_label_count = ner_label_count
        self.soft_prompt_len = soft_prompt_len
        self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, encoder_hidden_size).normal_(0, 0.02))
        self.prompt_proj_to_llama = nn.Linear(encoder_hidden_size, self.llama_hidden_size)
        if lora_cfg is not None:
            self.llama = get_peft_model(self.llama, lora_cfg)
        self.ner_head = nn.Linear(encoder_hidden_size, ner_label_count)
        self.consist_loss_fn = consist_loss_fn or self.default_consist_loss

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        if hasattr(self.llama, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs is not None:
                    self.llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                else:
                    self.llama.gradient_checkpointing_enable()
            except TypeError:
                self.llama.gradient_checkpointing_enable()
        try:
            self.llama.config.use_cache = False
        except Exception:
            pass

    def gradient_checkpointing_disable(self):
        if hasattr(self.llama, "gradient_checkpointing_disable"):
            self.llama.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, prompt_embeds, soft_prompt=None, labels=None,
                ner_context_embeds=None, ner_gold_labels=None, entity_mask=None, summary_ner_labels=None):
        """Forward path for the original prompted architecture, safe with device_map='auto'.

        The sharded 70B LLaMA expects the input embeddings to start on the embedding
        device. The wrapper-owned modules are kept on that same entry device by
        prepare_wrapper_for_auto_device_map(). This avoids DataParallel/device-map
        conflicts while preserving the original Stage3 prompt + soft prompt design.
        """
        embed_device = self.llama.get_input_embeddings().weight.device
        proj_device = self.prompt_proj_to_llama.weight.device
        target_dtype = self.prompt_proj_to_llama.weight.dtype

        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)
        if labels is not None:
            labels = labels.to(embed_device)

        B = input_ids.shape[0]
        P = prompt_embeds.shape[1]
        S = self.soft_prompt_len

        prompt_embeds = prompt_embeds.to(device=proj_device, dtype=target_dtype)
        if soft_prompt is None:
            soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(device=proj_device, dtype=target_dtype)
        else:
            soft_prompt = soft_prompt.to(device=proj_device, dtype=target_dtype)

        prompt_proj = self.prompt_proj_to_llama(prompt_embeds).to(embed_device)
        soft_proj = self.prompt_proj_to_llama(soft_prompt).to(embed_device)
        input_embeds = self.llama.get_input_embeddings()(input_ids)

        decoder_embeds = torch.cat([prompt_proj, soft_proj, input_embeds], dim=1)
        full_mask = torch.cat([
            torch.ones((B, P + S), dtype=attention_mask.dtype, device=embed_device),
            attention_mask,
        ], dim=1)

        labels_padded = labels
        if labels is not None:
            pad_len = decoder_embeds.shape[1] - labels.shape[1]
            if pad_len > 0:
                labels_padded = torch.cat([labels.new_full((labels.shape[0], pad_len), -100), labels], dim=1)

        llama_out = self.llama(inputs_embeds=decoder_embeds, attention_mask=full_mask, labels=labels_padded, use_cache=False)
        loss = llama_out.loss

        ner_loss = None
        if ner_context_embeds is not None and ner_gold_labels is not None:
            ner_device = self.ner_head.weight.device
            ner_logits = self.ner_head(ner_context_embeds.to(device=ner_device, dtype=self.ner_head.weight.dtype))
            ner_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                ner_logits.view(-1, self.ner_label_count),
                ner_gold_labels.to(ner_device).view(-1),
            )
            ner_loss = ner_loss.to(loss.device)

        consist_loss = None
        if ner_gold_labels is not None and summary_ner_labels is not None:
            consist_loss = self.default_consist_loss(labels, summary_ner_labels.to(embed_device), ner_gold_labels.to(embed_device), entity_mask)
            consist_loss = consist_loss.to(loss.device)

        total_loss = loss
        if ner_loss is not None:
            total_loss = total_loss + 0.2 * ner_loss
        if consist_loss is not None:
            total_loss = total_loss + 0.2 * consist_loss
        return {"loss": total_loss, "main_loss": loss, "ner_loss": ner_loss, "consist_loss": consist_loss}

    def default_consist_loss(self, labels, summary_ner_labels, ner_gold_labels, entity_mask):
        dialogue_entities = set(ner_gold_labels[ner_gold_labels != -100].detach().cpu().numpy().tolist())
        mask = summary_ner_labels != -100
        if len(dialogue_entities) == 0:
            return mask.float().mean() * 0.0
        missing = mask & (~torch.isin(summary_ner_labels, torch.tensor(list(dialogue_entities), device=summary_ner_labels.device)))
        return missing.float().mean()

class MultiTaskSummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

class DataCollatorTruncate:
    def __init__(self, train_max_len=1536, eval_max_len=1024):
        self.train_max_len = train_max_len
        self.eval_max_len = eval_max_len

    def __call__(self, batch):
        out = {}
        keys = batch[0].keys()
        # Trainer does not tell collator train/eval mode. Use train limit to preserve learning; generation slices inside prediction_step.
        max_len = self.train_max_len
        for k in keys:
            if k in ("input_ids", "attention_mask", "labels"):
                seqs = []
                for b in batch:
                    v = b[k]
                    seqs.append(v if isinstance(v, torch.Tensor) else torch.tensor(v))
                pad_val = 0 if k != "labels" else -100
                tens = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_val)
                if tens.size(1) > max_len:
                    tens = tens[:, -max_len:]
                out[k] = tens
            elif k in ("prompt_embeds", "ner_context_embeds", "ner_gold_labels", "entity_mask", "summary_ner_labels"):
                vals = [b[k] for b in batch]
                if isinstance(vals[0], torch.Tensor):
                    try:
                        out[k] = torch.stack(vals, dim=0)
                    except Exception:
                        # Pad variable length tensors on dim 0.
                        out[k] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=-100 if "label" in k else 0)
                else:
                    out[k] = vals
            else:
                # Keep text metadata for debugging only; Trainer will ignore because remove_unused_columns=False.
                out[k] = [b[k] for b in batch]
        return out

class MemoryCleanCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        cleanup_memory()
    def on_prediction_end(self, args, state, control, **kwargs):
        cleanup_memory()

class StepwiseLRSchedulerCallback(TrainerCallback):
    def __init__(self, patience=2, lr_list=None, monitor="eval_loss"):
        self.patience = patience
        self.lr_list = lr_list or [5e-5, 2e-5, 1e-5]
        self.monitor = monitor
        self.best = None
        self.bad_epochs = 0
        self.lr_index = 0
    def on_evaluate(self, args, state, control, metrics=None, optimizer=None, **kwargs):
        if metrics is None or optimizer is None:
            return
        current = metrics.get(self.monitor)
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
                print(f"\n[StepwiseLR] Reducing LR to {new_lr}")
                for group in optimizer.param_groups:
                    group["lr"] = new_lr

# =============================================================================
# TRAINER
# =============================================================================

class JointMultiTaskTrainer(Trainer):
    def __init__(self, *args, tokenizer_ref=None, val_dialogues_ref=None, seed_output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_ref = tokenizer_ref
        self.val_dialogues_ref = val_dialogues_ref or []
        self.seed_output_dir = seed_output_dir
        self.best_rouge1 = None
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.sacrebleu = evaluate.load("sacrebleu")
        self.bertscore = None
        self.bleurt = None
        self.unieval = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_embeds = inputs.pop("prompt_embeds")
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            prompt_embeds=prompt_embeds,
            labels=inputs["labels"],
            ner_context_embeds=inputs.get("ner_context_embeds"),
            ner_gold_labels=inputs.get("ner_gold_labels"),
            entity_mask=inputs.get("entity_mask"),
            summary_ner_labels=inputs.get("summary_ner_labels"),
        )
        return (output["loss"], output) if return_outputs else output["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            # For device_map="auto", use the embedding device for input ids.
            embed_device = model.llama.get_input_embeddings().weight.device
            proj_device = model.prompt_proj_to_llama.weight.device
            target_dtype = model.prompt_proj_to_llama.weight.dtype

            input_ids = inputs["input_ids"].to(embed_device)
            attention_mask = inputs["attention_mask"].to(embed_device)
            prompt_embeds = inputs["prompt_embeds"].to(proj_device, dtype=target_dtype)
            labels = inputs.get("labels")
            labels_for_loss = labels.to(embed_device) if labels is not None else None

            B = input_ids.size(0)
            # Prompt projection happens on projection device, then moved to embed device for LLaMA.
            prompt_proj = model.prompt_proj_to_llama(prompt_embeds).to(embed_device)
            soft_prompt = model.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(proj_device, dtype=target_dtype)
            soft_proj = model.prompt_proj_to_llama(soft_prompt).to(embed_device)
            input_embeds = model.llama.get_input_embeddings()(input_ids)

            gen_inputs_embeds = torch.cat([prompt_proj, soft_proj, input_embeds], dim=1)
            prefix_len = prompt_proj.size(1) + soft_proj.size(1)
            gen_attn = torch.cat([
                torch.ones((B, prefix_len), device=embed_device, dtype=attention_mask.dtype),
                attention_mask,
            ], dim=1)

            gen_kwargs = dict(
                inputs_embeds=gen_inputs_embeds,
                attention_mask=gen_attn,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                min_new_tokens=GEN_MIN_NEW_TOKENS,
                repetition_penalty=GEN_REPETITION_PENALTY,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
                num_beams=GEN_NUM_BEAMS,
                do_sample=None,
                length_penalty=0.9,
                use_cache=True,
                eos_token_id=self.tokenizer_ref.eos_token_id,
                pad_token_id=self.tokenizer_ref.pad_token_id,
            )
            if GEN_DO_SAMPLE:
                gen_kwargs["temperature"] = GEN_TEMPERATURE
                gen_kwargs["top_p"] = GEN_TOP_P
            generated_tokens = model.llama.generate(**gen_kwargs)

            # Teacher-forced loss uses original full architecture.
            loss = None
            if labels is not None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_embeds=inputs["prompt_embeds"].to(proj_device, dtype=target_dtype),
                    labels=labels_for_loss,
                    ner_context_embeds=inputs.get("ner_context_embeds"),
                    ner_gold_labels=inputs.get("ner_gold_labels"),
                    entity_mask=inputs.get("entity_mask"),
                    summary_ner_labels=inputs.get("summary_ner_labels"),
                )
                loss = outputs["loss"].detach()
            return (loss, generated_tokens, labels)

    def _load_heavy_metrics(self):
        if ENABLE_BERTSCORE and self.bertscore is None:
            self.bertscore = evaluate.load("bertscore")
        if ENABLE_BLEURT and self.bleurt is None:
            try:
                self.bleurt = evaluate.load("bleurt", config_name="bleurt-20")
            except Exception as e:
                print(f"[Metric Warning] BLEURT unavailable: {e}")
                self.bleurt = False
        if ENABLE_UNIEVAL and self.unieval is None:
            if get_evaluator is not None:
                try:
                    self.unieval = get_evaluator("summarization")
                except Exception as e:
                    print(f"[Metric Warning] UniEval unavailable: {e}")
                    self.unieval = False
            else:
                self.unieval = False

    def compute_metrics_fn(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(labels, tuple):
            labels = labels[0]
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
        predictions = predictions.astype(np.int32)
        labels = labels.astype(np.int32)
        labels = np.where(labels == -100, self.tokenizer_ref.pad_token_id, labels)
        predictions = np.where(predictions == -100, self.tokenizer_ref.pad_token_id, predictions)

        pred_str = self.tokenizer_ref.batch_decode(predictions, skip_special_tokens=True)
        label_str = self.tokenizer_ref.batch_decode(labels, skip_special_tokens=True)
        pred_str = [postprocess_generated_note(p) for p in pred_str]
        label_str = [canonicalize_note(l.strip()) for l in label_str]

        results = self.rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
        meteor_result = self.meteor.compute(predictions=pred_str, references=label_str)
        sacrebleu_result = self.sacrebleu.compute(predictions=pred_str, references=[[r] for r in label_str])

        metrics = {
            "rouge1": float(results["rouge1"]),
            "rouge2": float(results["rouge2"]),
            "rougeL": float(results["rougeL"]),
            "rougeLsum": float(results.get("rougeLsum", results["rougeL"])),
            "meteor": float(meteor_result["meteor"]),
            "bleu": float(sacrebleu_result["score"]),
        }

        self._load_heavy_metrics()
        if ENABLE_BERTSCORE and self.bertscore not in (None, False):
            try:
                bs = self.bertscore.compute(predictions=pred_str, references=label_str, lang="en", device="cuda" if torch.cuda.is_available() else "cpu", batch_size=1)
                metrics["bertscore_f1"] = float(np.mean(bs["f1"]))
            except Exception as e:
                print(f"[Metric Warning] BERTScore failed: {e}")
        if ENABLE_BLEURT and self.bleurt not in (None, False):
            try:
                bl = self.bleurt.compute(predictions=pred_str, references=label_str)
                metrics["bleurt"] = float(np.mean(bl["scores"]))
            except Exception as e:
                print(f"[Metric Warning] BLEURT failed: {e}")
        if ENABLE_UNIEVAL and self.unieval not in (None, False) and convert_to_json is not None:
            try:
                data = convert_to_json(output_list=pred_str, src_list=self.val_dialogues_ref, ref_list=label_str)
                scores = self.unieval.evaluate(data, print_result=False)
                if isinstance(scores, list) and scores and isinstance(scores[0], dict):
                    metrics.update({f"unieval_{k}": float(np.mean([d[k] for d in scores])) for k in scores[0].keys()})
                elif isinstance(scores, dict):
                    metrics.update({f"unieval_{k}": float(v) for k, v in scores.items()})
            except Exception as e:
                print(f"[Metric Warning] UniEval failed: {e}")
        return metrics

    def _save_trainable(self, output_dir: str, metrics: Optional[Dict[str, Any]] = None):
        os.makedirs(output_dir, exist_ok=True)
        # Save LoRA adapter.
        try:
            self.model.llama.save_pretrained(os.path.join(output_dir, "lora_adapter"))
        except Exception as e:
            print(f"[Save Warning] LoRA adapter save failed: {e}")
        try:
            self.tokenizer_ref.save_pretrained(output_dir)
        except Exception:
            pass
        small_state = {
            "soft_prompt": self.model.soft_prompt.detach().cpu(),
            "prompt_proj_to_llama": self.model.prompt_proj_to_llama.state_dict(),
            "ner_head": self.model.ner_head.state_dict(),
            "config": {
                "prompt_len": PROMPT_LEN,
                "soft_prompt_len": SOFT_PROMPT_LEN,
                "encoder_hidden_size": self.model.encoder_hidden_size,
                "ner_label_count": self.model.ner_label_count,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_modules": LORA_TARGET_MODULES,
            }
        }
        torch.save(small_state, os.path.join(output_dir, "custom_trainable_modules.pt"))
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics or {}, f, indent=2)
        print(f"[SaveBest] Saved trainable checkpoint to {output_dir}")

    def _save_checkpoint(self, model, trial):
        # Avoid full 70B checkpoint writes. Save trainable-only best by ROUGE-1.
        last_eval = {}
        for log in reversed(self.state.log_history):
            if "eval_rouge1" in log:
                last_eval = log
                break
        r1 = last_eval.get("eval_rouge1")
        if r1 is None:
            print(f"[SaveBest] eval_rouge1 not found at step {self.state.global_step}; skipping save.")
            return
        if self.best_rouge1 is None or r1 > self.best_rouge1:
            self.best_rouge1 = float(r1)
            best_dir = os.path.join(self.seed_output_dir, "best_model_trainable")
            if os.path.isdir(best_dir):
                shutil.rmtree(best_dir, ignore_errors=True)
            self._save_trainable(best_dir, last_eval)
            self.state.best_metric = float(r1)
            self.state.best_model_checkpoint = best_dir
        else:
            print(f"[SaveBest] rouge1={float(r1):.6f} did not improve over best={self.best_rouge1:.6f}; skipping save.")

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or os.path.join(self.seed_output_dir, "final_trainable")
        self._save_trainable(output_dir, {"note": "final trainable-only save"})

# =============================================================================
# DATA / PREPROCESS
# =============================================================================

def ensure_stage1():
    required_any_weight = [
        os.path.join(STAGE1_MODEL_DIR, "model.safetensors"),
        os.path.join(STAGE1_MODEL_DIR, "pytorch_model.bin"),
    ]
    has_config = os.path.exists(os.path.join(STAGE1_MODEL_DIR, "config.json"))
    has_weight = any(os.path.exists(p) for p in required_any_weight)
    if has_config and has_weight:
        print(f"NER model found at {STAGE1_MODEL_DIR}. Skipping NER training.")
        return
    if os.path.exists(STAGE1_SCRIPT):
        print(f"NER model not found at {STAGE1_MODEL_DIR}. Running NER training: {STAGE1_SCRIPT}")
        subprocess.run(["python", STAGE1_SCRIPT], check=True)
    else:
        raise FileNotFoundError(
            f"Stage1 model was not found at {STAGE1_MODEL_DIR} and Stage1 training script was not found. "
            f"Expected one of {required_any_weight} plus config.json. Checked script: {STAGE1_SCRIPT}"
        )


def load_or_preprocess(tokenizer):
    ensure_stage1()
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    train_dialogues = train_df["dialogue"].astype(str).tolist()
    train_notes = train_df["note"].astype(str).tolist()
    val_dialogues = val_df["dialogue"].astype(str).tolist()
    val_notes = val_df["note"].astype(str).tolist()

    print(f"Loaded {len(train_dialogues)} training dialogues and {len(val_dialogues)} validation dialogues.")
    os.makedirs(os.path.dirname(CACHE_TRAIN), exist_ok=True)
    if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_VAL):
        print("Loading preprocessed data from original cache paths...")
        train_data_list = torch.load(CACHE_TRAIN, map_location="cpu")
        val_data_list = torch.load(CACHE_VAL, map_location="cpu")
    else:
        print("Preprocessing with original Stage1/2/3 pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        stage1_model = AutoModelForTokenClassification.from_pretrained(STAGE1_MODEL_DIR).to(device)
        stage1_tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL_DIR)
        enc_h_size = stage1_model.config.hidden_size
        entity_label_count = stage1_model.config.num_labels
        stage2_encoder = BiasedAttentionEncoder(BACKBONE_NAME, enc_h_size, entity_label_count, bias_strength=3.0).to(device)
        stage3_proj = PromptProjection(hidden_size=enc_h_size, prompt_len=PROMPT_LEN, mode="attentive").to(device)
        train_data_list = preprocess_pipeline(train_dialogues, train_notes, stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj, tokenizer, device)
        val_data_list = preprocess_pipeline(val_dialogues, val_notes, stage1_model, stage1_tokenizer, stage2_encoder, stage3_proj, tokenizer, device)
        torch.save(train_data_list, CACHE_TRAIN)
        torch.save(val_data_list, CACHE_VAL)
        del stage1_model, stage2_encoder, stage3_proj
        cleanup_memory()
        print(f"Saved caches:\n  {CACHE_TRAIN}\n  {CACHE_VAL}")

    # Hidden size from Stage1 config.
    s1 = AutoModelForTokenClassification.from_pretrained(STAGE1_MODEL_DIR)
    enc_h_size = s1.config.hidden_size
    del s1
    cleanup_memory()
    return train_data_list, val_data_list, train_dialogues, val_dialogues, enc_h_size



def prepare_wrapper_for_auto_device_map(model: nn.Module):
    """Prevent Trainer from wrapping a device_map='auto' 70B model in DataParallel.

    The base LLaMA is already sharded across GPUs by Accelerate using device_map='auto'.
    Hugging Face Trainer may otherwise see multiple visible GPUs and wrap the whole
    wrapper in torch.nn.DataParallel, which breaks the device map and causes errors
    such as: Expected tensors on cuda:0 and cuda:1.

    We also place the custom small modules (soft prompt, prompt projection, NER head)
    on the LLaMA embedding device so inputs_embeds are assembled on the same device
    used to enter the sharded LLaMA.
    """
    llama = getattr(model, "llama", None)
    for obj in [model, llama, getattr(llama, "base_model", None), getattr(llama, "model", None)]:
        if obj is not None:
            try:
                obj.is_parallelizable = True
                obj.model_parallel = True
            except Exception:
                pass

    try:
        entry_device = model.llama.get_input_embeddings().weight.device
    except Exception:
        entry_device = next(model.parameters()).device

    # Move only the wrapper-owned trainable modules. Do not move the sharded LLaMA.
    with torch.no_grad():
        model.soft_prompt.data = model.soft_prompt.data.to(entry_device)
    model.prompt_proj_to_llama.to(entry_device)
    model.ner_head.to(entry_device)

    # Extra hints used by some Trainer/Accelerate versions.
    try:
        model.device = entry_device
    except Exception:
        pass
    try:
        model.hf_device_map = getattr(model.llama, "hf_device_map", None)
    except Exception:
        pass

    print(f"[DeviceMapFix] Trainer DataParallel disabled: is_parallelizable={getattr(model, 'is_parallelizable', None)}, model_parallel={getattr(model, 'model_parallel', None)}")
    print(f"[DeviceMapFix] Wrapper trainable modules placed on entry device: {entry_device}")
    if hasattr(model.llama, "hf_device_map"):
        try:
            print(f"[DeviceMapFix] LLaMA hf_device_map detected with {len(model.llama.hf_device_map)} entries.")
        except Exception:
            pass
    return model

# =============================================================================
# MODEL LOADING / TRAINING
# =============================================================================

def load_base_model_and_tokenizer():
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        pass
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    setattr(cfg, "attn_implementation", attn_impl)
    setattr(cfg, "_attn_implementation", attn_impl)
    cfg.use_cache = False

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    n_gpus = torch.cuda.device_count()
    gpu_cap = os.getenv("GPU_MAX_MEM", "46GiB")
    cpu_cap = os.getenv("CPU_MAX_MEM", "180GiB")
    max_mem = {i: gpu_cap for i in range(n_gpus)}
    max_mem["cpu"] = cpu_cap
    print(f"Loading {MODEL_NAME} in 4-bit QLoRA with device_map=auto over {n_gpus} visible GPUs...")
    print(f"max_memory={max_mem}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=cfg,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map="auto",
        max_memory=max_mem,
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.config.use_cache = False
    return base_model, tokenizer, compute_dtype


def train_one_seed(seed: int, shared_data):
    set_all_seeds(seed)
    train_data_list, val_data_list, train_dialogues, val_dialogues, enc_h_size = shared_data
    seed_outdir = os.path.join(MULTISEED_OUT_DIR, f"seed_{seed}")
    ckpt_dir = os.path.join(seed_outdir, "checkpoints")
    os.makedirs(seed_outdir, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Training seed {seed} -> {seed_outdir}")
    print("=" * 80)

    base_model, tokenizer, compute_dtype = load_base_model_and_tokenizer()
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = PromptedLlamaMultiTask(
        base_model=base_model,
        prompt_len=PROMPT_LEN,
        encoder_hidden_size=enc_h_size,
        ner_label_count=NER_LABEL_COUNT,
        soft_prompt_len=SOFT_PROMPT_LEN,
        lora_cfg=lora_cfg,
    )
    model = prepare_wrapper_for_auto_device_map(model)
    print_trainable_parameters(model)

    args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        metric_for_best_model="rouge1",
        greater_is_better=True,
        load_best_model_at_end=False,
        logging_steps=1,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        optim="paged_adamw_8bit",
    )

    trainer = JointMultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=MultiTaskSummarizationDataset(train_data_list),
        eval_dataset=MultiTaskSummarizationDataset(val_data_list),
        data_collator=DataCollatorTruncate(TRAIN_MAX_LEN, EVAL_MAX_LEN),
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            StepwiseLRSchedulerCallback(patience=2, lr_list=[LEARNING_RATE, 2e-5, 1e-5], monitor="eval_loss"),
            MemoryCleanCallback(),
        ],
        compute_metrics=None,
        tokenizer_ref=tokenizer,
        val_dialogues_ref=val_dialogues,
        seed_output_dir=seed_outdir,
    )
    trainer.compute_metrics = trainer.compute_metrics_fn

    print(f"[HParams] lr={LEARNING_RATE}, epochs={NUM_EPOCHS}, grad_accum={GRAD_ACCUM}, lora_r={LORA_R}, targets={LORA_TARGET_MODULES}")
    print(f"[Generation] max_new={GEN_MAX_NEW_TOKENS}, min_new={GEN_MIN_NEW_TOKENS}, beams={GEN_NUM_BEAMS}, sample={GEN_DO_SAMPLE}")
    trainer.train()

    final_dir = os.path.join(seed_outdir, "final_trainable")
    trainer.save_model(final_dir)

    eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
    best_log = max(eval_logs, key=lambda x: x.get("eval_rouge1", -1)) if eval_logs else {}
    best_metrics = {k.replace("eval_", ""): v for k, v in best_log.items()}
    with open(os.path.join(seed_outdir, "training_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    with open(os.path.join(seed_outdir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"[Seed {seed}] Best metrics: {best_metrics}")
    # Remove HF checkpoints; trainable saves are in best_model_trainable/final_trainable.
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    del trainer, model, base_model, tokenizer
    cleanup_memory()
    return best_metrics

# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--force_reprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(MULTISEED_OUT_DIR, exist_ok=True)
    if args.force_reprocess:
        for p in [CACHE_TRAIN, CACHE_VAL]:
            if os.path.exists(p):
                os.remove(p)
    # Tokenizer first for preprocessing.
    tmp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tmp_tokenizer.pad_token is None:
        tmp_tokenizer.pad_token = tmp_tokenizer.eos_token
    shared_data = load_or_preprocess(tmp_tokenizer)
    del tmp_tokenizer
    cleanup_memory()

    all_metrics: Dict[int, Dict[str, Any]] = {}
    cache_path = os.path.join(MULTISEED_OUT_DIR, "seed_results_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                all_metrics = {int(k): v for k, v in json.load(f).items()}
        except Exception:
            all_metrics = {}

    for seed in args.seeds:
        metrics = train_one_seed(seed, shared_data)
        all_metrics[seed] = metrics
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in all_metrics.items()}, f, indent=2)

    # Summary over completed seeds.
    completed = sorted(all_metrics.keys())
    metric_keys = sorted({k for m in all_metrics.values() for k in m.keys() if k not in {"epoch", "step"} and not k.endswith("runtime")})
    summary = {}
    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY — original QLoRA 70B architecture")
    print("=" * 80)
    print(f"{'Metric':<28} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)
    for k in metric_keys:
        vals = []
        for s in completed:
            try:
                vals.append(float(all_metrics[s][k]))
            except Exception:
                pass
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        summary[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max()), "values": {str(s): all_metrics[s].get(k) for s in completed}}
        print(f"{k:<28} {arr.mean():>10.4f} {arr.std():>10.4f} {arr.min():>10.4f} {arr.max():>10.4f}")

    out = {"model": MODEL_NAME, "seeds": completed, "per_seed": {str(k): v for k, v in all_metrics.items()}, "summary": summary}
    with open(os.path.join(MULTISEED_OUT_DIR, "multiseed_train_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Done] Saved summary to {os.path.join(MULTISEED_OUT_DIR, 'multiseed_train_summary.json')}")

if __name__ == "__main__":
    main()
