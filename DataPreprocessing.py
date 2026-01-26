# Data Preprocessing

import torch

SECTIONS = [
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

def build_structured_prompt(dialogue):
    instruction = (
        "[INSTRUCTION]\n"
        "Given the following doctor-patient dialogue, generate a clinical note in the following format:\n\n"
    )
    structure = "\n\n".join(SECTIONS) + "\n\n[DOCTOR-PATIENT DIALOGUE]\n"
    note_marker = "\n\n[NOTE]\n"
    return instruction + structure + dialogue + note_marker

def chunk_tokenize(text, tokenizer, max_length):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    token_chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=False) for chunk in token_chunks]

def preprocess_pipeline(
    dialogues, notes,  # lists of raw text and reference summaries
    stage1_model, stage1_tokenizer,
    stage2_encoder, stage3_proj,
    llama_tokenizer,
    device="cuda"
):
    data_list = []
    stage1_model.to(device)
    stage2_encoder.to(device)
    stage3_proj.to(device)

    BERT_MAXLEN = 512
    prompt_len = stage3_proj.prompt_len
    LLAMA_MAX_LEN = 3072

    for dialog, note in zip(dialogues, notes):
        prompt = build_structured_prompt(dialog)

        # --------- PATCH: Truncate note to fit input length -----------
        prompt_ids = llama_tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        note_ids = llama_tokenizer(note, return_tensors="pt")["input_ids"].squeeze(0)
        total_len = len(prompt_ids) + len(note_ids)
        if total_len > LLAMA_MAX_LEN:
            print(f"[WARN] Example too long: prompt={len(prompt_ids)}, note={len(note_ids)} ({total_len} > {LLAMA_MAX_LEN}). Truncating note.")
            note_ids = note_ids[:(LLAMA_MAX_LEN - len(prompt_ids))]
            note = llama_tokenizer.decode(note_ids, skip_special_tokens=True)
        # -------------------------------------------------------------

        full_input = prompt + note  

        
        enc = llama_tokenizer(
            full_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=LLAMA_MAX_LEN,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # 3. Find where note begins in tokens (to mask labels)
        prompt_token_ids = llama_tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        prompt_token_count = prompt_token_ids.shape[0]
        labels = input_ids.clone()
        labels[:prompt_token_count] = -100

        # -------------- FIXED PROMPT EMBED CHUNKING ------------------
        # 1. Tokenize prompt to tokens, chunk by BERT_MAXLEN
        prompt_tokens = stage1_tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
        prompt_chunks = [prompt_tokens[i:i+BERT_MAXLEN] for i in range(0, len(prompt_tokens), BERT_MAXLEN)]
        all_prompt_embeds = []
        all_contextual_embeds = []
        all_bias_masks = []
        all_entity_pred_ids = []

        for chunk_tok in prompt_chunks:
            # Pad every chunk to BERT_MAXLEN
            cur_len = chunk_tok.size(0)
            if cur_len < BERT_MAXLEN:
                pad = torch.full((BERT_MAXLEN - cur_len,), stage1_tokenizer.pad_token_id, device=chunk_tok.device)
                chunk_tok = torch.cat([chunk_tok, pad], dim=0)
            # **DECODE BACK TO STRING for tokenizer input**
            chunk_str = stage1_tokenizer.decode(chunk_tok, skip_special_tokens=False)
            s1_inputs = stage1_tokenizer(
                chunk_str,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=BERT_MAXLEN,
            )
            s1_inputs = {k: v.to(device) for k, v in s1_inputs.items()}
            with torch.no_grad():
                ner_outputs = stage1_model(**s1_inputs)
                entity_pred_ids = torch.argmax(ner_outputs.logits, dim=-1)  # [1, L]
            with torch.no_grad():
                contextual_embeds, bias_mask, entity_pred_ids = stage2_encoder(
                    input_ids=s1_inputs["input_ids"],
                    attention_mask=s1_inputs["attention_mask"],
                    entity_pred_ids=entity_pred_ids
                )  # [1, L, H], [1, L]
            with torch.no_grad():
                prompt_embeds = stage3_proj(contextual_embeds, bias_mask)  # [1, P, H]
            all_prompt_embeds.append(prompt_embeds)
            all_contextual_embeds.append(contextual_embeds)
            all_bias_masks.append(bias_mask)
            all_entity_pred_ids.append(entity_pred_ids)

        # ------------- END CHUNK PATCH ------------------------------
        # Now all_prompt_embeds is a list of [1, P, H] for each chunk

        prompt_embeds = torch.cat(all_prompt_embeds, dim=1)  # [1, N*P, H]
        if prompt_embeds.shape[1] > prompt_len:
            prompt_embeds = prompt_embeds[:, :prompt_len, :]
        elif prompt_embeds.shape[1] < prompt_len:
            pad = torch.zeros((1, prompt_len - prompt_embeds.shape[1], prompt_embeds.shape[2]), device=prompt_embeds.device)
            prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)

        ner_context_embeds = all_contextual_embeds[0].squeeze(0).cpu()
        ner_gold_labels = all_entity_pred_ids[0].squeeze(0).cpu()
        entity_mask = all_bias_masks[0].squeeze(0).cpu()

        entry = {
            "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "labels": labels.cpu(),
            "ner_context_embeds": ner_context_embeds,
            "ner_gold_labels": ner_gold_labels,
            "entity_mask": entity_mask,
            "target_text": note,
            "input_text": prompt,
        }
        data_list.append(entry)
    return data_list


