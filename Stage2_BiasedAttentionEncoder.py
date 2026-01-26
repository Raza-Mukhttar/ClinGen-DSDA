# DSDA

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm

# === Enhanced Dual-Stage Dynamic Attention Encoder ===
class BiasedAttentionEncoder(nn.Module):
    def __init__(self, backbone_name, hidden_size, entity_label_count, bias_strength=3.0, top_k_ratio=0.25):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        self.entity_embed = nn.Embedding(entity_label_count, hidden_size)
        self.bias_strength = bias_strength
        self.score_proj = nn.Linear(hidden_size, 1)  # Stage 1: Importance scoring
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)  # Stage 2: Focused attention
        self.top_k_ratio = top_k_ratio

    def forward(self, input_ids, attention_mask, entity_pred_ids):
        # Stage 1: Contextual encoding + Entity embedding bias
        enc_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = enc_outputs.last_hidden_state  
        entity_emb = self.entity_embed(entity_pred_ids)  
        h = h + entity_emb  # Inject entity information

        # Stage 2: Importance Scoring
        scores = self.score_proj(h).squeeze(-1) 
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)  

        # Stage 3: Select Top-K important tokens
        B, L, H = h.size()
        K = max(1, int(self.top_k_ratio * L))
        topk_vals, topk_indices = torch.topk(weights, K, dim=1)  

        # Stage 4: Focused Multi-Head Attention on Top-K
        batch_indices = torch.arange(B, device=input_ids.device).unsqueeze(1).expand(-1, K)
        selected_tokens = h[batch_indices, topk_indices]  
        attn_output, _ = self.attn(selected_tokens, selected_tokens, selected_tokens) 

        # Stage 5: Merge focused representation back into original sequence
        h_updated = h.clone()
        h_updated[batch_indices, topk_indices] = attn_output  # Replace top-K slots

        # Stage 6: Bias mask from NER predictions
        bias_mask = (entity_pred_ids > 0).float() * self.bias_strength  

        return h_updated, bias_mask, entity_pred_ids

# === Dataset Wrapper ===
class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.dialogues[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

# === Stage 1 + Stage 2 Batch Inference ===
@torch.no_grad()
def run_stage1_stage2_batch(
    dataloader, stage1_model, stage2_encoder, device
):
    all_embeds, all_biases, all_entity_ids = [], [], []
    stage1_model.eval()
    stage2_encoder.eval()
    for batch in tqdm(dataloader, desc="Stage 1+2"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            ner_outputs = stage1_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            ner_outputs = stage1_model(input_ids=input_ids, attention_mask=attention_mask)
        entity_pred_ids = torch.argmax(ner_outputs.logits, dim=-1)
        contextual_embeds, bias_mask = stage2_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_pred_ids=entity_pred_ids
        )
        all_embeds.append(contextual_embeds.cpu())
        all_biases.append(bias_mask.cpu())
        all_entity_ids.append(entity_pred_ids.cpu())
    return (
        torch.cat(all_embeds, dim=0),
        torch.cat(all_biases, dim=0),
        torch.cat(all_entity_ids, dim=0)
    )

# === Preprocess and Save Outputs ===
def preprocess_and_save(
    split_name, csv_path, outdir,
    stage1_model, stage1_tokenizer, stage2_encoder, device, batch_size=8
):
    df = pd.read_csv(csv_path)
    dialogues = df["dialogue"].astype(str).tolist()
    dataset = DialogueDataset(dialogues, stage1_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Processing {split_name} set: {len(dialogues)} samples...")
    embeds, bias_masks, entity_ids = run_stage1_stage2_batch(
        dataloader, stage1_model, stage2_encoder, device
    )
    os.makedirs(outdir, exist_ok=True)
    torch.save(
        {
            "contextual_embeds": embeds,
            "bias_masks": bias_masks,
            "entity_pred_ids": entity_ids,
            "dialogues": dialogues,
            "meta": {"split": split_name}
        },
        os.path.join(outdir, f"{split_name}_stage2_outputs.pt")
    )
    print(f"Saved: {os.path.join(outdir, f'{split_name}_stage2_outputs.pt')}")







