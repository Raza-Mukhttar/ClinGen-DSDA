

import re
from typing import Dict, List, Tuple

import torch

# -----------------------------------------------------------------------------
# One canonical section list used for prompt, target note canonicalization, and
# later evaluation. Keep this aligned with the ACI-Bench note format.
# -----------------------------------------------------------------------------
SECTIONS = [
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "REVIEW OF SYSTEMS",
    "VITALS",
    "PHYSICAL EXAMINATION",
    "RESULTS",
    "ASSESSMENT AND PLAN",
    "INSTRUCTIONS",
]

# Alternative / noisy section names that may occur in raw reference notes.
SECTION_ALIASES = {
    "CC": "CHIEF COMPLAINT",
    "CHIEF COMPLAINT": "CHIEF COMPLAINT",
    "HPI": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "HISTORY": "HISTORY OF PRESENT ILLNESS",
    "REVIEW OF SYSTEMS": "REVIEW OF SYSTEMS",
    "ROS": "REVIEW OF SYSTEMS",
    "VITAL": "VITALS",
    "VITALS": "VITALS",
    "VITAL SIGNS": "VITALS",
    "PHYSICAL EXAM": "PHYSICAL EXAMINATION",
    "PHYSICAL EXAMINATION": "PHYSICAL EXAMINATION",
    "EXAM": "PHYSICAL EXAMINATION",
    "RESULT": "RESULTS",
    "RESULTS": "RESULTS",
    "LABS": "RESULTS",
    "LABORATORY": "RESULTS",
    "LABORATORY RESULTS": "RESULTS",
    "ASSESSMENT": "ASSESSMENT AND PLAN",
    "PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT AND PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT/PLAN": "ASSESSMENT AND PLAN",
    "A/P": "ASSESSMENT AND PLAN",
    "INSTRUCTION": "INSTRUCTIONS",
    "INSTRUCTIONS": "INSTRUCTIONS",
    "PATIENT INSTRUCTIONS": "INSTRUCTIONS",
}

HEADER_PATTERN = re.compile(
    r"(?im)^\s*(?:\[)?(CC|CHIEF COMPLAINT|HPI|HISTORY OF PRESENT ILLNESS|HISTORY|"
    r"REVIEW OF SYSTEMS|ROS|VITAL|VITALS|VITAL SIGNS|PHYSICAL EXAM|PHYSICAL EXAMINATION|EXAM|"
    r"RESULT|RESULTS|LABS|LABORATORY|LABORATORY RESULTS|ASSESSMENT AND PLAN|ASSESSMENT/PLAN|"
    r"ASSESSMENT|PLAN|A/P|PATIENT INSTRUCTIONS|INSTRUCTION|INSTRUCTIONS)(?:\])?\s*:?\s*$"
)


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving section-like line breaks."""
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _canonical_header(raw_header: str):
    key = raw_header.strip().upper().replace("[", "").replace("]", "").replace(":", "")
    key = re.sub(r"\s+", " ", key)
    return SECTION_ALIASES.get(key)


def extract_sections(text: str, section_headers: List[str] = SECTIONS) -> Dict[str, str]:
    """Extract note sections robustly from raw clinical-note text.

    The function supports headings with/without brackets/colon and maps common
    aliases such as HPI, ROS, Physical Exam, Assessment, and Plan to the target
    canonical section list.
    """
    text = _clean_text(text)
    sections = {h: "" for h in section_headers}
    current = None
    buffer = []

    def flush():
        nonlocal buffer, current
        if current is not None and buffer:
            content = "\n".join(buffer).strip()
            if content:
                if sections[current]:
                    sections[current] = (sections[current] + "\n" + content).strip()
                else:
                    sections[current] = content
        buffer = []

    lines = text.split("\n")
    saw_header = False
    fallback_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current is not None and buffer and buffer[-1] != "":
                buffer.append("")
            continue

        m = HEADER_PATTERN.match(line_stripped)
        if m:
            mapped = _canonical_header(m.group(1))
            if mapped in sections:
                flush()
                current = mapped
                saw_header = True
                continue

        # Inline heading like "HISTORY OF PRESENT ILLNESS: patient ..."
        inline = re.match(
            r"(?i)^\s*(CHIEF COMPLAINT|HPI|HISTORY OF PRESENT ILLNESS|REVIEW OF SYSTEMS|ROS|"
            r"VITALS|VITAL SIGNS|PHYSICAL EXAM|PHYSICAL EXAMINATION|RESULTS|ASSESSMENT AND PLAN|"
            r"ASSESSMENT|PLAN|INSTRUCTIONS)\s*:\s*(.*)$",
            line_stripped,
        )
        if inline:
            mapped = _canonical_header(inline.group(1))
            if mapped in sections:
                flush()
                current = mapped
                saw_header = True
                rest = inline.group(2).strip()
                if rest:
                    buffer.append(rest)
                continue

        if current is None:
            fallback_lines.append(line_stripped)
        else:
            buffer.append(line_stripped)

    flush()

    # If no sections were detected, place raw content into HPI so the model still
    # learns clinically useful text rather than an empty template.
    if not saw_header and fallback_lines:
        sections["HISTORY OF PRESENT ILLNESS"] = "\n".join(fallback_lines).strip()

    # If source has separate ASSESSMENT and PLAN, aliases already merge both into
    # ASSESSMENT AND PLAN. Remove duplicate blank lines.
    for k, v in sections.items():
        v = re.sub(r"\n{3,}", "\n\n", v).strip()
        sections[k] = v
    return sections


def canonicalize_note(text: str, section_headers: List[str] = SECTIONS) -> str:
    """Canonical target note used for model supervision and ROUGE alignment."""
    sections = extract_sections(text, section_headers)
    blocks = []
    for header in section_headers:
        content = sections.get(header, "").strip()
        # Do not write "None" because it hurts ROUGE and teaches hallucinated text.
        blocks.append(f"{header}\n\n{content}".strip())
    return "\n\n".join(blocks).strip() + "\n"


def build_structured_prompt(dialogue: str) -> str:
    """Short prompt designed to reduce instruction echo and improve ROUGE.

    We use plain headings with colons in the instruction template but the target
    note is canonicalized without colons. The model learns the target form from
    labels while the prompt remains compact.
    """
    dialogue = _clean_text(dialogue)
    section_template = "\n".join([f"{s}:" for s in SECTIONS])
    return (
        "Generate a structured clinical note from the doctor-patient dialogue.\n"
        "Use exactly the section headings below. Do not copy the instruction.\n"
        "Include only facts stated in the dialogue. Leave unknown sections blank.\n\n"
        f"{section_template}\n\n"
        "[DIALOGUE]\n"
        f"{dialogue}\n\n"
        "[NOTE]\n"
    )


def _tokenize_no_special(tokenizer, text: str):
    """Tokenize without adding duplicate BOS/EOS where supported."""
    try:
        return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    except TypeError:
        return tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)


def _decode_tokens(tokenizer, ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _fit_prompt_and_note(
    prompt: str,
    note: str,
    llama_tokenizer,
    llama_max_len: int,
    min_note_tokens: int = 768,
) -> Tuple[str, str]:
    """Fit prompt + note into LLaMA max length, prioritizing target-note retention.

    The old preprocessing truncated the note first, which can directly suppress
    ROUGE by training on incomplete references. This function preserves as much
    target note as possible and truncates the dialogue/prompt from the left when
    necessary. Left truncation preserves the most recent dialogue context, which
    often contains assessment/plan instructions.
    """
    prompt_ids = _tokenize_no_special(llama_tokenizer, prompt)
    note_ids = _tokenize_no_special(llama_tokenizer, note)

    if prompt_ids.numel() + note_ids.numel() <= llama_max_len:
        return prompt, note

    # Keep note unless it alone exceeds the budget.
    if note_ids.numel() > min_note_tokens:
        max_note_len = min(note_ids.numel(), max(min_note_tokens, llama_max_len // 2))
    else:
        max_note_len = note_ids.numel()

    # If note is extremely long, cap it but keep section starts.
    note_ids = note_ids[: min(note_ids.numel(), max_note_len)]
    remaining_for_prompt = max(32, llama_max_len - int(note_ids.numel()))

    if prompt_ids.numel() > remaining_for_prompt:
        # Keep tail of prompt/dialogue because final recommendations are often near end.
        prompt_ids = prompt_ids[-remaining_for_prompt:]
        prompt = _decode_tokens(llama_tokenizer, prompt_ids)
        # Ensure prompt still ends with [NOTE]. If left-truncation removed the boundary,
        # rebuild a compact prompt using the retained tail as dialogue.
        if "[NOTE]" not in prompt:
            prompt = build_structured_prompt(prompt)
            prompt_ids = _tokenize_no_special(llama_tokenizer, prompt)
            if prompt_ids.numel() > remaining_for_prompt:
                prompt_ids = prompt_ids[-remaining_for_prompt:]
                prompt = _decode_tokens(llama_tokenizer, prompt_ids)
                if "[NOTE]" not in prompt:
                    prompt = prompt + "\n\n[NOTE]\n"

    note = _decode_tokens(llama_tokenizer, note_ids)
    return prompt, note


def _pad_or_truncate_1d(tensor: torch.Tensor, max_len: int, pad_value: int) -> torch.Tensor:
    tensor = tensor[:max_len]
    if tensor.numel() < max_len:
        pad = torch.full((max_len - tensor.numel(),), pad_value, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor


def _make_llama_features(prompt: str, note: str, llama_tokenizer, llama_max_len: int):
    """Create input_ids, attention_mask, labels with exact prompt masking."""
    # Build full sequence manually so prompt_token_count is exact.
    prompt_ids = _tokenize_no_special(llama_tokenizer, prompt)
    note_ids = _tokenize_no_special(llama_tokenizer, note)

    # Add EOS at the end of the note if available; this helps generation stop.
    eos_id = getattr(llama_tokenizer, "eos_token_id", None)
    if eos_id is not None:
        note_ids = torch.cat([note_ids, torch.tensor([eos_id], dtype=note_ids.dtype)], dim=0)

    full_ids = torch.cat([prompt_ids, note_ids], dim=0)
    prompt_token_count = int(prompt_ids.numel())

    if full_ids.numel() > llama_max_len:
        # Final safety: preserve the prompt boundary and truncate note tail.
        available_note = max(1, llama_max_len - prompt_token_count)
        if available_note <= 1:
            # Prompt is too long; keep prompt tail and at least one note token.
            prompt_ids = prompt_ids[-(llama_max_len - 1):]
            prompt_token_count = int(prompt_ids.numel())
            note_ids = note_ids[:1]
        else:
            note_ids = note_ids[:available_note]
        full_ids = torch.cat([prompt_ids, note_ids], dim=0)

    pad_id = llama_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = llama_tokenizer.eos_token_id

    seq_len = int(full_ids.numel())
    input_ids = _pad_or_truncate_1d(full_ids.cpu(), llama_max_len, int(pad_id))
    attention_mask = torch.zeros(llama_max_len, dtype=torch.long)
    attention_mask[:seq_len] = 1

    labels = input_ids.clone()
    labels[:prompt_token_count] = -100
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels, prompt_token_count


def _stage_prompt_text(dialogue: str) -> str:
    """Text passed to Stage1/2/3 for entity-biased soft prompts.

    Use dialogue only. Passing instruction/template tokens contaminates entity
    embeddings and wastes Stage3 prompt capacity.
    """
    return _clean_text(dialogue)


def _extract_stage_prompt_embeds(
    dialogue: str,
    stage1_model,
    stage1_tokenizer,
    stage2_encoder,
    stage3_proj,
    device: str,
    bert_max_len: int = 512,
):
    prompt_len = stage3_proj.prompt_len
    stage_text = _stage_prompt_text(dialogue)
    tokens = stage1_tokenizer(stage_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    if tokens.numel() == 0:
        tokens = torch.tensor([stage1_tokenizer.cls_token_id or stage1_tokenizer.pad_token_id])

    chunks = [tokens[i : i + bert_max_len] for i in range(0, tokens.numel(), bert_max_len)]
    all_prompt_embeds = []
    all_contextual_embeds = []
    all_bias_masks = []
    all_entity_pred_ids = []

    for chunk_tok in chunks:
        chunk_tok = chunk_tok.cpu()
        cur_len = int(chunk_tok.numel())
        if cur_len < bert_max_len:
            pad = torch.full((bert_max_len - cur_len,), stage1_tokenizer.pad_token_id, dtype=chunk_tok.dtype)
            chunk_tok = torch.cat([chunk_tok, pad], dim=0)
        chunk_str = stage1_tokenizer.decode(chunk_tok, skip_special_tokens=False)
        s1_inputs = stage1_tokenizer(
            chunk_str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=bert_max_len,
        )
        s1_inputs = {k: v.to(device) for k, v in s1_inputs.items()}

        with torch.no_grad():
            ner_outputs = stage1_model(**s1_inputs)
            entity_pred_ids = torch.argmax(ner_outputs.logits, dim=-1)
            contextual_embeds, bias_mask, entity_pred_ids = stage2_encoder(
                input_ids=s1_inputs["input_ids"],
                attention_mask=s1_inputs["attention_mask"],
                entity_pred_ids=entity_pred_ids,
            )
            prompt_embeds = stage3_proj(contextual_embeds, bias_mask)

        all_prompt_embeds.append(prompt_embeds.detach())
        all_contextual_embeds.append(contextual_embeds.detach())
        all_bias_masks.append(bias_mask.detach())
        all_entity_pred_ids.append(entity_pred_ids.detach())

    # Concatenate prompt tokens from all chunks, then keep first prompt_len.
    # If your Stage3 projection ranks tokens by attention internally, this captures
    # more than the first chunk while preserving fixed shape.
    prompt_embeds = torch.cat(all_prompt_embeds, dim=1)
    if prompt_embeds.shape[1] > prompt_len:
        prompt_embeds = prompt_embeds[:, :prompt_len, :]
    elif prompt_embeds.shape[1] < prompt_len:
        pad = torch.zeros(
            (1, prompt_len - prompt_embeds.shape[1], prompt_embeds.shape[2]),
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )
        prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)

    # Auxiliary NER supervision can use concatenated chunks but is capped to 512
    # for memory and collator compatibility.
    ner_context = torch.cat(all_contextual_embeds, dim=1)[:, :bert_max_len, :]
    ner_labels = torch.cat(all_entity_pred_ids, dim=1)[:, :bert_max_len]
    entity_mask = torch.cat(all_bias_masks, dim=1)[:, :bert_max_len]

    # Pad auxiliary tensors if dialogue shorter than 512.
    cur_len = ner_labels.shape[1]
    if cur_len < bert_max_len:
        h = ner_context.shape[-1]
        ctx_pad = torch.zeros((1, bert_max_len - cur_len, h), device=ner_context.device, dtype=ner_context.dtype)
        lab_pad = torch.full((1, bert_max_len - cur_len), -100, device=ner_labels.device, dtype=ner_labels.dtype)
        mask_pad = torch.zeros((1, bert_max_len - cur_len), device=entity_mask.device, dtype=entity_mask.dtype)
        ner_context = torch.cat([ner_context, ctx_pad], dim=1)
        ner_labels = torch.cat([ner_labels, lab_pad], dim=1)
        entity_mask = torch.cat([entity_mask, mask_pad], dim=1)

    return (
        prompt_embeds.squeeze(0).cpu(),
        ner_context.squeeze(0).cpu(),
        ner_labels.squeeze(0).cpu(),
        entity_mask.squeeze(0).cpu(),
    )


def preprocess_pipeline(
    dialogues,
    notes,
    stage1_model,
    stage1_tokenizer,
    stage2_encoder,
    stage3_proj,
    llama_tokenizer,
    device="cuda",
):
    """Preprocess ACI-Bench dialogues/notes for the Stage4 QLoRA trainer.

    Returns a list of dictionaries containing the same keys expected by your
    current Original QLoRA Multi Device training script:
      - prompt_embeds
      - input_ids
      - attention_mask
      - labels
      - ner_context_embeds
      - ner_gold_labels
      - entity_mask
      - target_text
      - input_text
    """
    data_list = []

    stage1_model.to(device)
    stage2_encoder.to(device)
    stage3_proj.to(device)
    stage1_model.eval()
    stage2_encoder.eval()
    stage3_proj.eval()

    bert_max_len = 512
    llama_max_len = 3072

    if llama_tokenizer.pad_token_id is None and llama_tokenizer.eos_token_id is not None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    for idx, (dialogue, raw_note) in enumerate(zip(dialogues, notes)):
        dialogue = _clean_text(dialogue)
        note = canonicalize_note(raw_note)
        prompt = build_structured_prompt(dialogue)
        prompt, note = _fit_prompt_and_note(
            prompt=prompt,
            note=note,
            llama_tokenizer=llama_tokenizer,
            llama_max_len=llama_max_len,
            min_note_tokens=768,
        )

        input_ids, attention_mask, labels, prompt_token_count = _make_llama_features(
            prompt=prompt,
            note=note,
            llama_tokenizer=llama_tokenizer,
            llama_max_len=llama_max_len,
        )

        prompt_embeds, ner_context_embeds, ner_gold_labels, entity_mask = _extract_stage_prompt_embeds(
            dialogue=dialogue,
            stage1_model=stage1_model,
            stage1_tokenizer=stage1_tokenizer,
            stage2_encoder=stage2_encoder,
            stage3_proj=stage3_proj,
            device=device,
            bert_max_len=bert_max_len,
        )

        entry = {
            "prompt_embeds": prompt_embeds,
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "labels": labels.cpu(),
            "ner_context_embeds": ner_context_embeds,
            "ner_gold_labels": ner_gold_labels,
            "entity_mask": entity_mask,
            "target_text": note,
            "input_text": prompt,
            "prompt_token_count": int(prompt_token_count),
        }
        data_list.append(entry)

        if (idx + 1) % 25 == 0:
            print(f"[Preprocess] processed {idx + 1}/{len(dialogues)} examples")

    return data_list
