# =============================================================================
# =============================================================================

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Keep TensorFlow/BLEURT off the GPUs (same as training).
try:
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
except Exception:
    pass

import gc
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# -----------------------------------------------------------------------------
# Paths / constants — identical defaults to Stage4.py, all overridable via env.
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("CLINGEN_ROOT", "/home/project")
if os.path.isdir(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stage 2 / Stage 3 / preprocessing imports — same fallbacks as Stage4.py.
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

MODEL_NAME = os.environ.get("LLAMA70B_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")
STAGE1_MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "stage1_outputs", "final_model")

# Architecture defaults (match Stage4.py HYPERPARAMETERS block).
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "16"))
SOFT_PROMPT_LEN = int(os.environ.get("SOFT_PROMPT_LEN", "16"))
NER_LABEL_COUNT = int(os.environ.get("NER_LABEL_COUNT", "5"))
BACKBONE_NAME = os.environ.get("STAGE2_BACKBONE", "emilyalsentzer/Bio_ClinicalBERT")

LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = os.environ.get("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")

# Generation config — identical to Stage4.prediction_step defaults.
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "512"))
GEN_MIN_NEW_TOKENS = int(os.environ.get("GEN_MIN_NEW_TOKENS", "128"))
GEN_NUM_BEAMS = int(os.environ.get("GEN_NUM_BEAMS", "4"))
GEN_DO_SAMPLE = os.environ.get("GEN_DO_SAMPLE", "false").lower() == "true"
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "0.9"))
GEN_REPETITION_PENALTY = float(os.environ.get("GEN_REPETITION_PENALTY", "1.10"))
GEN_NO_REPEAT_NGRAM = int(os.environ.get("GEN_NO_REPEAT_NGRAM", "3"))

SECTION_HEADERS = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS", "VITALS", "RESULTS", "INSTRUCTIONS"]
SECTION_HEADERS_FULL = [
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS", "VITALS",
    "PHYSICAL EXAMINATION", "RESULTS", "ASSESSMENT", "PLAN", "ASSESSMENT AND PLAN", "INSTRUCTIONS",
]


# -----------------------------------------------------------------------------
# Utils (copied verbatim from Stage4.py so decoding/scoring stays identical).
# -----------------------------------------------------------------------------
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def set_all_seeds(seed: int):
    import random
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


def extract_sections(text: str, section_headers=SECTION_HEADERS) -> Dict[str, str]:
    sections: Dict[str, str] = {}
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
    note_match = re.search(r"\[NOTE\]", generated_text, re.IGNORECASE)
    if note_match:
        after_note = generated_text[note_match.end():]
    else:
        m = re.search(r"(?:CHIEF COMPLAINT|HISTORY OF PRESENT ILLNESS|REVIEW OF SYSTEMS|VITALS|RESULTS|INSTRUCTIONS)", generated_text, re.IGNORECASE)
        after_note = generated_text[m.start():] if m else generated_text
    end_match = re.search(r"\[END OF NOTE\]|\[END NOTE\]|\[END INSTRUCTION\]|\[INSTRUCTION\]", after_note, re.IGNORECASE)
    note_body = after_note[:end_match.start()] if end_match else after_note
    note_body = normalize_section_headers(note_body)
    note_body = re.sub(r"\n{3,}", "\n\n", note_body)
    return note_body.strip("\n\r\t ")


# -----------------------------------------------------------------------------
# Model wrapper — same class as training, generation-only path added.
# -----------------------------------------------------------------------------
class PromptedLlamaMultiTask(nn.Module):
    def __init__(self, base_model, prompt_len, encoder_hidden_size, ner_label_count,
                 soft_prompt_len=8, lora_cfg=None):
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

    @torch.no_grad()
    def generate_note(self, input_ids, attention_mask, prompt_embeds, tokenizer):
        """Reproduces Stage4.prediction_step generation exactly."""
        self.eval()
        embed_device = self.llama.get_input_embeddings().weight.device
        proj_device = self.prompt_proj_to_llama.weight.device
        target_dtype = self.prompt_proj_to_llama.weight.dtype

        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)
        prompt_embeds = prompt_embeds.to(proj_device, dtype=target_dtype)

        B = input_ids.size(0)
        prompt_proj = self.prompt_proj_to_llama(prompt_embeds).to(embed_device)
        soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(proj_device, dtype=target_dtype)
        soft_proj = self.prompt_proj_to_llama(soft_prompt).to(embed_device)
        input_embeds = self.llama.get_input_embeddings()(input_ids)

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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if GEN_DO_SAMPLE:
            gen_kwargs["temperature"] = GEN_TEMPERATURE
            gen_kwargs["top_p"] = GEN_TOP_P
        generated_tokens = self.llama.generate(**gen_kwargs)
        return generated_tokens


# -----------------------------------------------------------------------------
# Base model + tokenizer (identical 4-bit QLoRA / device_map=auto to training).
# -----------------------------------------------------------------------------
def load_base_model_and_tokenizer(for_generation: bool = True):
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        pass
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    setattr(cfg, "attn_implementation", attn_impl)
    setattr(cfg, "_attn_implementation", attn_impl)
    # For pure inference we WANT the cache on.
    cfg.use_cache = bool(for_generation)

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

    n_gpus = max(torch.cuda.device_count(), 1)
    gpu_cap = os.getenv("GPU_MAX_MEM", "46GiB")
    cpu_cap = os.getenv("CPU_MAX_MEM", "180GiB")
    max_mem = {i: gpu_cap for i in range(n_gpus)}
    max_mem["cpu"] = cpu_cap
    print(f"[Load] {MODEL_NAME} 4-bit QLoRA, device_map=auto over {n_gpus} GPU(s). max_memory={max_mem}")

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
    # prepare_model_for_kbit_training is fine for eval too; it just fixes norms/casts.
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    base_model.config.use_cache = bool(for_generation)
    return base_model, tokenizer, compute_dtype


def _get_stage1_hidden_size() -> int:
    s1 = AutoModelForTokenClassification.from_pretrained(STAGE1_MODEL_DIR)
    enc_h = s1.config.hidden_size
    del s1
    cleanup_memory()
    return enc_h


def load_trained_model(checkpoint_dir: str):
    """Rebuild PromptedLlamaMultiTask and load a Stage4 trainable-only checkpoint.

    checkpoint_dir must contain:
        lora_adapter/                    (PEFT adapter dir)
        custom_trainable_modules.pt      (soft_prompt, prompt_proj_to_llama, ner_head, config)
    """
    custom_path = os.path.join(checkpoint_dir, "custom_trainable_modules.pt")
    adapter_dir = os.path.join(checkpoint_dir, "lora_adapter")
    if not os.path.exists(custom_path):
        raise FileNotFoundError(f"Missing {custom_path}")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Missing LoRA adapter dir {adapter_dir}")

    state = torch.load(custom_path, map_location="cpu")
    saved_cfg = state.get("config", {})
    enc_h = int(saved_cfg.get("encoder_hidden_size") or _get_stage1_hidden_size())
    prompt_len = int(saved_cfg.get("prompt_len", PROMPT_LEN))
    soft_len = int(saved_cfg.get("soft_prompt_len", SOFT_PROMPT_LEN))
    ner_labels = int(saved_cfg.get("ner_label_count", NER_LABEL_COUNT))

    base_model, tokenizer, _ = load_base_model_and_tokenizer(for_generation=True)

    # Build wrapper WITHOUT a fresh LoraConfig, then attach the saved adapter.
    model = PromptedLlamaMultiTask(
        base_model=base_model,
        prompt_len=prompt_len,
        encoder_hidden_size=enc_h,
        ner_label_count=ner_labels,
        soft_prompt_len=soft_len,
        lora_cfg=None,
    )
    # Attach trained LoRA adapter to the base LLaMA.
    model.llama = PeftModel.from_pretrained(model.llama, adapter_dir, is_trainable=False)

    # Place wrapper-owned modules on the LLaMA embedding (entry) device.
    entry_device = model.llama.get_input_embeddings().weight.device
    with torch.no_grad():
        model.soft_prompt.data = state["soft_prompt"].to(entry_device)
    model.prompt_proj_to_llama.load_state_dict(state["prompt_proj_to_llama"])
    model.ner_head.load_state_dict(state["ner_head"])
    model.prompt_proj_to_llama.to(entry_device)
    model.ner_head.to(entry_device)
    try:
        model.hf_device_map = getattr(model.llama, "hf_device_map", None)
    except Exception:
        pass

    model.eval()
    print(f"[Load] Restored trainable modules from {checkpoint_dir} "
          f"(enc_h={enc_h}, prompt_len={prompt_len}, soft_len={soft_len}, ner_labels={ner_labels}).")
    return model, tokenizer, enc_h


# -----------------------------------------------------------------------------
# Preprocessing — build the per-example dicts with prompt_embeds, exactly like
# Stage4.load_or_preprocess() does for the validation split.
# -----------------------------------------------------------------------------
def build_examples(dialogues: List[str], notes: Optional[List[str]], tokenizer, enc_h_size: int):
    """Run the Stage1/2/3 pipeline to produce data dicts (prompt_embeds etc.).

    If `notes` is None (pure inference), passes empty strings so the pipeline can
    still tokenize; label-dependent fields are simply ignored during generation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if notes is None:
        notes = ["" for _ in dialogues]

    stage1_model = AutoModelForTokenClassification.from_pretrained(STAGE1_MODEL_DIR).to(device)
    stage1_tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL_DIR)
    entity_label_count = stage1_model.config.num_labels
    stage2_encoder = BiasedAttentionEncoder(BACKBONE_NAME, enc_h_size, entity_label_count, bias_strength=3.0).to(device)
    stage3_proj = PromptProjection(hidden_size=enc_h_size, prompt_len=PROMPT_LEN, mode="attentive").to(device)

    data_list = preprocess_pipeline(
        dialogues, notes, stage1_model, stage1_tokenizer,
        stage2_encoder, stage3_proj, tokenizer, device,
    )
    del stage1_model, stage2_encoder, stage3_proj
    cleanup_memory()
    return data_list


def collate_eval(batch, eval_max_len: int = 1536):
    """Minimal collator for generation: pad input_ids/attention_mask, stack prompt_embeds.

    Mirrors DataCollatorTruncate but keeps only the fields generation needs and
    slices to the LEFT context window (keep the most recent tokens) like training.
    """
    out: Dict[str, Any] = {}
    for k in ("input_ids", "attention_mask"):
        seqs = [b[k] if isinstance(b[k], torch.Tensor) else torch.tensor(b[k]) for b in batch]
        pad_val = 0
        tens = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_val)
        if tens.size(1) > eval_max_len:
            tens = tens[:, -eval_max_len:]
        out[k] = tens
    pe = [b["prompt_embeds"] for b in batch]
    out["prompt_embeds"] = torch.stack(pe, dim=0) if isinstance(pe[0], torch.Tensor) else pe
    # Carry references/ids through if present.
    for k in ("note", "reference", "dialogue", "id"):
        if k in batch[0]:
            out[k] = [b.get(k) for b in batch]
    return out
