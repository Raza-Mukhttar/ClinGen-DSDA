
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
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "4")
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
from Endocrinology.Stage2_BiasedAttentionEncoder import BiasedAttentionEncoder
from Endocrinology.Stage3_PromptInjection import PromptProjection
from EMRV1.EMRV2.Endocrinology.DataPreprocessingEndo import preprocess_pipeline

# ------------------------------ Stage1 bootstrap ------------------------------
ner_model_dir = "/Endocrinology/outputs/stage1_outputs/final_model/model.safetensors"
if not os.path.exists(ner_model_dir):
    import subprocess
    print("NER model not found. Running NER training...")
    subprocess.run(["python", "/Endocrinology/Stage1_NER_Module.py"], check=True)
else:
    print("NER model found. Skipping NER training.")

# ------------------------------ Utilities ------------------------------
SECTION_HEADERS = [
    "Demographics",
    "Presenting Illness",
    "History of Presenting Illness",
    "Past Medical History",
    "Surgical History",
    "Family History",
    "Social History",
    "Allergy History",
    "Medication History",
    "Dietary History",
    "Review of Systems",
    "Physical Exam Findings",
    "Labs and Imaging",
    "Assessment",
    "Plan"]

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
    "Demographics",
    "Presenting Illness",
    "History of Presenting Illness",
    "Past Medical History",
    "Surgical History",
    "Family History",
    "Social History",
    "Allergy History",
    "Medication History",
    "Dietary History",
    "Review of Systems",
    "Physical Exam Findings",
    "Labs and Imaging",
    "Assessment",
    "Plan"
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
sys.path.append('/UniEval')
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
    print("Loading endocrinology datasets from Hugging Face...")
    from datasets import load_dataset

    train_data = load_dataset("starfishdata/endocrinology_structured_notes_1500")["train"]
    val_data   = load_dataset("starfishdata/endocrinology_structured_notes_eval")["train"]
    test_data  = load_dataset("starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper")["train"]

    # Convert to DataFrames for consistency
    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)
    test_df  = pd.DataFrame(test_data)

    # Extract dialogues and corresponding clinical notes
    train_dialogues = [str(x) for x in train_df["transcript"]]
    val_dialogues   = [str(x) for x in val_df["transcript"]]
    test_dialogues  = [str(x) for x in test_df["transcript"]]

    train_notes = [str(x) for x in train_df["structured_note"]]
    val_notes   = [str(x) for x in val_df["structured_note"]]
    test_notes  = [str(x) for x in test_df["structured_note"]]

    
    outdir = "/Endocrinology/outputs/stage4_outputs"
    os.makedirs(outdir, exist_ok=True)

    print(f"✅ Loaded {len(train_dialogues)} training dialogues, "
          f"{len(val_dialogues)} validation dialogues, "
          f"{len(test_dialogues)} test dialogues.")

    # ---- Stage1/2/3 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_model_dir = "/Endocrinology/outputs/stage1_outputs/final_model"
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

    cache_train = "/Endocrinology/outputs/preprocessing_outputs/train70b_data_list.pt"
    cache_val   = "/Endocrinology/outputs/preprocessing_outputs/val70b_data_list.pt"
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
    ds_cfg_path = "/ds_config_zero2.json"
    deepspeed_cfg = load_and_patch_ds_config(ds_cfg_path, use_bf16=(compute_dtype == torch.bfloat16))

    # ---- Truncation limits (env-tunable)
    TRAIN_MAX_LEN = int(os.getenv("TRAIN_MAX_LEN", "1536"))
    EVAL_MAX_LEN  = int(os.getenv("EVAL_MAX_LEN",  "1024"))

    training_args = TrainingArguments(
        output_dir="/Endocrinology/outputs/model_training_outputs/llama70b_joint_summ_ckpt",
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
    # trainer.save_model("/Endocrinology/outputs/model_training_outputs/llama70b_joint_summ_ckpt/final")
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

