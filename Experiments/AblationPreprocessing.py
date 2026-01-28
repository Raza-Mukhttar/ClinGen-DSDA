import torch


def preprocess_pipeline(
    dialogues, notes,
    stage1_model, stage1_tokenizer,
    stage2_encoder, stage3_proj,
    llama_tokenizer,
    device="cuda",
    ablation=None
):
    if ablation is None:
        ablation = {}

    data_list = []
    stage1_model.to(device)
    stage2_encoder.to(device)
    stage3_proj.to(device)

    for dialog, note in zip(dialogues, notes):
        # --------- Stage 1: NER on dialogue ---------
        if ablation.get("skip_ner", False):
            
            s1_inputs = stage1_tokenizer(dialog, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            s1_inputs = {k: v.to(device) for k, v in s1_inputs.items()}

            
            with torch.no_grad():
                contextual_embeds = stage2_encoder.encoder(
                    input_ids=s1_inputs["input_ids"],
                    attention_mask=s1_inputs["attention_mask"]
                ).last_hidden_state  

            
            bias_mask = torch.ones_like(s1_inputs["input_ids"]) 
            entity_pred_ids = torch.zeros_like(s1_inputs["input_ids"]) 

        else:
            
            s1_inputs = stage1_tokenizer(dialog, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            s1_inputs = {k: v.to(device) for k, v in s1_inputs.items()}
            with torch.no_grad():
                ner_outputs = stage1_model(**s1_inputs)
                entity_pred_ids = torch.argmax(ner_outputs.logits, dim=-1)  # [1, L]
            with torch.no_grad():
                contextual_embeds, bias_mask, entity_pred_ids = stage2_encoder(
                    input_ids=s1_inputs["input_ids"],
                    attention_mask=s1_inputs["attention_mask"],
                    entity_pred_ids=entity_pred_ids
                )  

        # --------- Stage 3: Prompt Projection ---------
        if ablation.get("skip_prompt", False):
            prompt_embeds = torch.zeros(1, stage3_proj.prompt_len, contextual_embeds.shape[-1], device=device)
        else:
            with torch.no_grad():
                prompt_embeds = stage3_proj(contextual_embeds, bias_mask)  # [1, P, H]

        # --------- LLaMA Tokenization of Summary/Note ---------
        summary_inputs = llama_tokenizer(note, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = summary_inputs["input_ids"].squeeze(0).to(device)
        attention_mask = summary_inputs["attention_mask"].squeeze(0).to(device)

        # --------- Stage 1 NER on summary/note ---------
        if ablation.get("skip_summary_ner", False):
            summary_ner_labels = torch.zeros_like(input_ids)
        else:
            s1_sum_inputs = stage1_tokenizer(note, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            s1_sum_inputs = {k: v.to(device) for k, v in s1_sum_inputs.items()}
            with torch.no_grad():
                ner_sum_outputs = stage1_model(**s1_sum_inputs)
                summary_ner_labels = torch.argmax(ner_sum_outputs.logits, dim=-1).squeeze(0)
        summary_ner_labels[attention_mask == 0] = -100

        entry = {
            "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "labels": input_ids.cpu(),
            "ner_context_embeds": contextual_embeds.squeeze(0).cpu(),
            "ner_gold_labels": entity_pred_ids.squeeze(0).cpu(),
            "entity_mask": bias_mask.squeeze(0).cpu(),
            "summary_ner_labels": summary_ner_labels.cpu()
        }
        data_list.append(entry)
    return data_list
