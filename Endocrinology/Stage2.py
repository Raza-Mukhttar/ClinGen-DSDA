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










# Multiple Datasets
# file: stage2_biased_attention_encoder.py

# import os, torch, numpy as np
# from tqdm import tqdm
# from torch import nn
# from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
# from datasets import load_dataset

# # ===================== TOGGLES =====================
# USE_GENERAL_DOMAIN = True
# USE_HF_SAMSUM = True
# # ===================================================

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Paths
# STAGE1_DIR = "/home/raza/EMRV1/EMRV2/outputs/stage111_outputs/final_model"
# OUTDIR = "/home/raza/EMRV1/EMRV2/outputs/stage222_outputs"
# os.makedirs(OUTDIR, exist_ok=True)

# # ===================== ENTITY SETTINGS =====================
# if USE_GENERAL_DOMAIN:
#     ENTITY_TYPES = [
#         "PERSON","ORG","GPE","LOC","NORP","FAC","PRODUCT","EVENT","WORK_OF_ART",
#         "LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
#     ]
# else:
#     ENTITY_TYPES = ["DISEASE", "CHEMICAL"]

# LABEL_LIST = ["O"] + [f"{p}-{etype}" for etype in ENTITY_TYPES for p in ["B","I"]]
# label2id = {label:i for i,label in enumerate(LABEL_LIST)}
# id2label = {i:label for i,label in enumerate(LABEL_LIST)}
# ENTITY_LABEL_COUNT = len(LABEL_LIST)
# # ============================================================

# # ---------------- DATA LOADING ----------------
# def load_dialogues():
#     if USE_HF_SAMSUM:
#         ds = load_dataset("knkarthick/samsum")
#         all_texts = ds["train"]["dialogue"] + ds["validation"]["dialogue"] + ds["test"]["dialogue"]
#     else:
#         csv_path = "/path/to/your/dialogue_data.csv"
#         import pandas as pd
#         df = pd.read_csv(csv_path)
#         all_texts = df["dialogue"].astype(str).tolist()
#     return all_texts

# # ---------------- STAGE 2 MODEL ----------------
# class BiasedAttentionEncoder(nn.Module):
#     def __init__(self, base_model_name, entity_label_count=ENTITY_LABEL_COUNT, bias_strength=1.0, top_k_ratio=0.25):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(base_model_name)
#         hidden_size = self.encoder.config.hidden_size
#         self.entity_embed = nn.Embedding(entity_label_count, hidden_size)
#         self.linear_score = nn.Linear(hidden_size, 1)
#         self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
#         self.bias_strength = bias_strength
#         self.top_k_ratio = top_k_ratio

#     def forward(self, input_ids, attention_mask, entity_pred_ids):
#         h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         # replace ignore idx (-100) with 0 to avoid embed crash
#         safe_ids = entity_pred_ids.clone()
#         safe_ids[safe_ids < 0] = 0
#         entity_emb = self.entity_embed(safe_ids)
#         h_biased = h + entity_emb * self.bias_strength

#         scores = self.linear_score(h_biased).squeeze(-1)
#         scores = scores.masked_fill(attention_mask == 0, float("-inf"))
#         attn_weights = torch.softmax(scores, dim=-1)
#         topk_count = max(1, int(self.top_k_ratio * h.size(1)))
#         topk_vals, topk_idx = torch.topk(attn_weights, k=topk_count, dim=-1)

#         B, L, H = h.shape
#         bias_mask = (entity_pred_ids > 0).float() * self.bias_strength
#         out_embeds = torch.zeros_like(h)

#         for b in range(B):
#             idxs = topk_idx[b]
#             sel = h_biased[b, idxs, :].unsqueeze(0)
#             q = h_biased[b].unsqueeze(0)
#             attn_out, _ = self.attn(q, sel, sel, need_weights=False)
#             out_embeds[b] = attn_out[0]

#         return out_embeds, bias_mask, entity_pred_ids

# # ---------------- INFERENCE PIPELINE ----------------
# @torch.no_grad()
# def run_stage2():
#     print("Loading Stage 1 NER model...")
#     ner_model = AutoModelForTokenClassification.from_pretrained(STAGE1_DIR).to(DEVICE)
#     tokenizer = AutoTokenizer.from_pretrained(STAGE1_DIR, use_fast=True)
#     ner_model.eval()

#     print("Loading dialogues...")
#     dialogues = load_dialogues()
#     encoder = BiasedAttentionEncoder(base_model_name=ner_model.config._name_or_path).to(DEVICE)
#     encoder.eval()

#     contextual_embeds, bias_masks, entity_ids = [], [], []

#     for text in tqdm(dialogues, desc="Stage 2 processing"):
#         enc = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(DEVICE)
#         outputs = ner_model(**enc)
#         preds = torch.argmax(outputs.logits, dim=-1)
#         contextual, bias_mask, ent_ids = encoder(enc["input_ids"], enc["attention_mask"], preds)
#         contextual_embeds.append(contextual.cpu())
#         bias_masks.append(bias_mask.cpu())
#         entity_ids.append(ent_ids.cpu())

#     torch.save({"contextual_embeds": contextual_embeds,
#                 "bias_masks": bias_masks,
#                 "entity_ids": entity_ids}, os.path.join(OUTDIR, "stage222_outputs.pt"))
#     print("Stage 2 complete — embeddings saved to:", OUTDIR)


# if __name__ == "__main__":
#     run_stage2()
