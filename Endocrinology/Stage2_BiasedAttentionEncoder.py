# DSDA

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

# === Enhanced Dual-Stage Dynamic Attention Encoder ===
# actual start
# class BiasedAttentionEncoder(nn.Module):
#     def __init__(self, backbone_name, hidden_size, entity_label_count, bias_strength=3.0, top_k_ratio=0.25):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(backbone_name)
#         self.entity_embed = nn.Embedding(entity_label_count, hidden_size)
#         self.bias_strength = bias_strength
#         self.score_proj = nn.Linear(hidden_size, 1)  # Stage 1: Importance scoring
#         self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)  # Stage 2: Focused attention
#         self.top_k_ratio = top_k_ratio

#     def forward(self, input_ids, attention_mask, entity_pred_ids):
#         # Stage 1: Contextual encoding + Entity embedding bias
#         enc_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         h = enc_outputs.last_hidden_state  
#         entity_emb = self.entity_embed(entity_pred_ids)  
#         h = h + entity_emb  # Inject entity information

#         # Stage 2: Importance Scoring
#         scores = self.score_proj(h).squeeze(-1) 
#         scores = scores.masked_fill(attention_mask == 0, float('-inf'))
#         weights = torch.softmax(scores, dim=-1)  

#         # Stage 3: Select Top-K important tokens
#         B, L, H = h.size()
#         K = max(1, int(self.top_k_ratio * L))
#         topk_vals, topk_indices = torch.topk(weights, K, dim=1)  

#         # Stage 4: Focused Multi-Head Attention on Top-K
#         batch_indices = torch.arange(B, device=input_ids.device).unsqueeze(1).expand(-1, K)
#         selected_tokens = h[batch_indices, topk_indices]  
#         attn_output, _ = self.attn(selected_tokens, selected_tokens, selected_tokens) 

#         # Stage 5: Merge focused representation back into original sequence
#         h_updated = h.clone()
#         h_updated[batch_indices, topk_indices] = attn_output  # Replace top-K slots

#         # Stage 6: Bias mask from NER predictions
#         bias_mask = (entity_pred_ids > 0).float() * self.bias_strength  

#         return h_updated, bias_mask, entity_pred_ids
# actual end

#updated dsda start
class BiasedAttentionEncoder(nn.Module):
    """
    Gated-Biased DSDA Encoder
    - Guarantees inclusion of mandatory entity tokens (hard gate)
    - Adds positive bias to those tokens inside the focused Multi-Head Attention
    - Keeps the original interface: returns (h_updated, bias_mask, entity_pred_ids)

    Args:
        backbone_name: HF backbone for contextual token embeddings (e.g., ClinicalBERT).
        hidden_size:   Hidden size H (should match backbone).
        entity_label_count: number of NER labels from Stage-1 model.
        bias_strength: float added to attention logits for critical (mandatory) tokens (soft bias).
        top_k_ratio:   ratio of tokens selected for focused attention (K = int(top_k_ratio * L)).
        mandatory_label_ids: list of label IDs that are considered "must-include".
                             If None or empty, gating is disabled.
        max_k:         optional cap on K to bound compute for long sequences.
    """
    def __init__(
        self,
        backbone_name: str,
        hidden_size: int,
        entity_label_count: int,
        bias_strength: float = 2.0,
        top_k_ratio: float = 0.25,
        mandatory_label_ids: Optional[List[int]] = None,
        max_k: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        self.entity_embed = nn.Embedding(entity_label_count, hidden_size)

        self.bias_strength = float(bias_strength)
        self.top_k_ratio = float(top_k_ratio)
        self.max_k = max_k

        self.score_proj = nn.Linear(hidden_size, 1)                 # Stage 1: importance scoring
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, # Stage 2: focused MHA
                                          batch_first=True)

        # store mandatory IDs as a non-persistent buffer for easy device moves
        if mandatory_label_ids:
            mand = torch.tensor(mandatory_label_ids, dtype=torch.long)
            self.register_buffer("mandatory_ids", mand, persistent=False)
        else:
            self.mandatory_ids = None

    def _build_gate_mask(self, entity_pred_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns Boolean gate mask [B, L] where True marks mandatory tokens.
        If no mandatory ids configured, returns all False.
        """
        if self.mandatory_ids is None or self.mandatory_ids.numel() == 0:
            return torch.zeros_like(entity_pred_ids, dtype=torch.bool)
        # compare against each mandatory id and OR together
        gate = torch.zeros_like(entity_pred_ids, dtype=torch.bool)
        for lid in self.mandatory_ids:
            gate |= (entity_pred_ids == lid)
        return gate

    def forward(self, input_ids, attention_mask, entity_pred_ids):
        """
        Inputs:
            input_ids:        [B, L]
            attention_mask:   [B, L] (1 for real tokens, 0 for pad)
            entity_pred_ids:  [B, L] (token-level label ids from Stage-1)

        Returns:
            h_updated:        [B, L, H]  (focused + biased embeddings written back)
            bias_mask:        [B, L]     (float mask, >0 at entity tokens; can be used downstream)
            entity_pred_ids:  [B, L]     (passthrough for convenience/debug)
        """
        B, L = input_ids.size()
        H = self.encoder.config.hidden_size  # should equal provided hidden_size

        # ---- contextual encoding + entity injection ----
        enc_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = enc_outputs.last_hidden_state  # [B, L, H]

        ent_emb = self.entity_embed(entity_pred_ids)  # [B, L, H]
        h = h + ent_emb

        # ---- importance scoring ----
        scores = self.score_proj(h).squeeze(-1)                 # [B, L]
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                 # [B, L]

        # ---- hard gating (mandatory inclusion) ----
        gate_mask = self._build_gate_mask(entity_pred_ids)      # [B, L] bool
        # how many to select total
        K = max(1, int(self.top_k_ratio * L))
        if self.max_k is not None:
            K = min(K, int(self.max_k))

        # precompute per-batch selected indices (union of mandatory ∪ topK of the rest)
        selected_idx = []
        for b in range(B):
            g_idx = torch.nonzero(gate_mask[b], as_tuple=False).squeeze(-1)  # [G] or []
            G = g_idx.numel()

            # pick remaining by weights among non-mandatory, non-pad
            mask_rest = (~gate_mask[b]) & (attention_mask[b] == 1)
            rest_scores = scores[b].clone()
            rest_scores[~mask_rest] = float("-inf")
            need = max(0, K - G)
            if need > 0:
                topk = torch.topk(rest_scores, k=min(need, mask_rest.sum().item()), dim=0).indices
                sel = torch.cat([g_idx, topk], dim=0)
            else:
                # if G >= K, keep all mandatory (or take first K if you want to cap)
                sel = g_idx if G > 0 else torch.topk(weights[b], k=K, dim=0).indices

            # ensure unique + sorted for stable gather
            sel = torch.unique(sel, sorted=True)
            selected_idx.append(sel)

        # pad selection to same K' length across batch
        Kp = max(s.numel() for s in selected_idx)
        # guard: if a batch has 0 real tokens (shouldn't happen), force Kp>=1
        Kp = max(1, Kp)
        idx_mat = torch.full((B, Kp), fill_value=0, dtype=torch.long, device=input_ids.device)
        valid_mask = torch.zeros((B, Kp), dtype=torch.bool, device=input_ids.device)
        for b, s in enumerate(selected_idx):
            k = s.numel()
            idx_mat[b, :k] = s
            valid_mask[b, :k] = True

        # ---- gather selected tokens ----
        batch_arange = torch.arange(B, device=input_ids.device).unsqueeze(-1)
        selected_tokens = h[batch_arange, idx_mat]  # [B, K', H]

        # ---- attention bias for keys (positive pushes prob. mass toward mandatory) ----
        # Build key-wise bias vector (per batch), then broadcast to [B, K', K'] as additive mask
        key_bias = torch.zeros((B, Kp), dtype=h.dtype, device=h.device)  # [B, K']
        # mark which selected positions are mandatory
        selected_is_mand = gate_mask[batch_arange, idx_mat]              # [B, K'] bool
        key_bias[selected_is_mand] = self.bias_strength

        # Build additive mask to add to attention logits BEFORE softmax
        # PyTorch MHA supports float attn_mask added to logits (>= 1.11/1.12).
        attn_bias = key_bias.unsqueeze(1).expand(B, Kp, Kp).contiguous()  # [B, Kq, Kk] (= [B, K', K'])

        # Key padding mask (False=keep, True=pad) for positions we padded in idx_mat
        key_padding = ~valid_mask  # [B, K'] bool

        # ---- focused multi-head attention ----
        # Note: attn_mask is added to logits; key_padding_mask masks padded keys.
        num_heads = self.attn.num_heads
        # Repeat attn_bias across heads -> [B*num_heads, K', K']
        attn_bias_expanded = attn_bias.repeat_interleave(num_heads, dim=0)
        attn_out, _ = self.attn(
            selected_tokens, selected_tokens, selected_tokens,
            attn_mask=attn_bias_expanded,                # additive bias
            key_padding_mask=key_padding,       # True -> mask out
            need_weights=False
        )  # [B, K', H]

        # ---- write back to original sequence ----
        h_updated = h.clone()
        h_updated[batch_arange.expand(-1, Kp), idx_mat] = attn_out

        # ---- bias mask for downstream (float >0 at entity tokens) ----
        # Keep your previous behavior for compatibility, but you can also OR (gate_mask) later if you want.
        bias_mask = (entity_pred_ids > 0).float() * self.bias_strength  # [B, L]

        return h_updated, bias_mask, entity_pred_ids

#updated dsda end





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
        contextual_embeds, bias_mask, entity_pred_ids = stage2_encoder(
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
    stage1_model, stage1_tokenizer, stage2_encoder, device, batch_size=1
):
    df = pd.read_csv(csv_path)
    dialogues = df["transcript"].astype(str).tolist()
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






if __name__ == "__main__":
    # -------- LOAD Stage 1 Model --------
    stage1_model_dir = "/home/raza/EMRV1/EMRV2/Endocrinology/outputs/stage1_outputs/final_model"
    stage1_model = AutoModelForTokenClassification.from_pretrained(stage1_model_dir)
    stage1_tokenizer = AutoTokenizer.from_pretrained(stage1_model_dir)

    # -------- SETUP Stage 2 Encoder -----
    backbone = "emilyalsentzer/Bio_ClinicalBERT"
    hidden_size = stage1_model.config.hidden_size
    entity_label_count = stage1_model.config.num_labels
    stage2_encoder = BiasedAttentionEncoder(
        backbone_name=backbone,
        hidden_size=hidden_size,
        entity_label_count=entity_label_count,
        bias_strength=3.0
    )

    # --------- DEVICE ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_model = stage1_model.to(device)
    stage2_encoder = stage2_encoder.to(device)

    # --------- PATHS -----------
    print("Loading endocrinology datasets from Hugging Face...")
    from datasets import load_dataset
    train_data = load_dataset("starfishdata/endocrinology_structured_notes_1500")["train"]
    val_data   = load_dataset("starfishdata/endocrinology_structured_notes_eval")["train"]
    test_data  = load_dataset("starfishdata/endocrinology_structured_notes_aci_bench_eval_whisper")["train"]

    # Convert to DataFrames (for consistency and debugging)
    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)
    test_df  = pd.DataFrame(test_data)

    # Extract doctor–patient dialogues
    train_dialogues = [str(x) for x in train_df["transcript"]]
    val_dialogues   = [str(x) for x in val_df["transcript"]]
    test_dialogues  = [str(x) for x in test_df["transcript"]]

    outdir = "/home/raza/EMRV1/EMRV2/Endocrinology/outputs/stage2_outputs"
    os.makedirs(outdir, exist_ok=True)

    print(f"Loaded {len(train_dialogues)} training dialogues, "
          f"{len(val_dialogues)} validation dialogues, "
          f"{len(test_dialogues)} test dialogues.")

    # datadir = "/home/raza/work/raza/EMRV1/datasets"
    # outdir = "./stage2_outputs"
    # splits = {
    #     "train": os.path.join(datadir, "ACI-TaskB-TrainingSet.csv"),
    #     "val":   os.path.join(datadir, "ACI-TaskB-ValidationSet.csv"),
    #     "test":  os.path.join(datadir, "ACI-TaskB-TestingSet.csv"),
    # }

    # --------- RUN ON ALL SPLITS (no parameter updates here) ---------
# Wrap and process splits
    print("\nProcessing splits with Stage-1 + Stage-2 models...\n")

    # --------- RUN STAGE 1 + STAGE 2 ON EACH SPLIT ---------
    splits = {
        "train": train_dialogues,
        "val": val_dialogues,
        "test": test_dialogues,
    }

    for split_name, dialogues in splits.items():
        dataset = DialogueDataset(dialogues, stage1_tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        print(f"Processing {split_name} split ({len(dialogues)} samples)...")
        embeds, bias_masks, entity_ids = run_stage1_stage2_batch(
            dataloader, stage1_model, stage2_encoder, device
        )

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
        print(f"Saved {split_name} split outputs to {outdir}/{split_name}_stage2_outputs.pt\n")

    print("\n✅ All splits processed successfully! Ready for LLM training/evaluation.\n")

    # for split_name, csv_path in splits.items():
    #     preprocess_and_save(
    #         split_name, csv_path, outdir,
    #         stage1_model, stage1_tokenizer, stage2_encoder, device,
    #         batch_size=8,
    #     )

    # print("\nAll splits processed! Ready for LLM training/evaluation.\n")














# # Stage 2 — Gated-Biased DSDA preprocessing (embeds + bias masks)
# # Stage 2 — Gated-Biased DSDA preprocessing (embeds + bias masks)
# import os
# import json
# from typing import List, Optional

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
# import pandas as pd
# from tqdm import tqdm


# # ---------------- Gated-Biased DSDA Encoder ----------------
# class BiasedAttentionEncoder(nn.Module):
#     """
#     Gated-Biased DSDA Encoder
#     - Guarantees inclusion of mandatory entity tokens (hard gate)
#     - Adds positive bias to those tokens inside the focused Multi-Head Attention
#     - Returns: (h_updated, bias_mask, entity_pred_ids)
#     """

#     def __init__(
#         self,
#         backbone_name: str,
#         hidden_size: int,
#         entity_label_count: int,
#         bias_strength: float = 2.0,
#         top_k_ratio: float = 0.25,
#         mandatory_label_ids: Optional[List[int]] = None,
#         max_k: Optional[int] = None,
#     ):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(backbone_name)
#         self.entity_embed = nn.Embedding(entity_label_count, hidden_size)

#         self.bias_strength = float(bias_strength)
#         self.top_k_ratio = float(top_k_ratio)
#         self.max_k = max_k

#         self.score_proj = nn.Linear(hidden_size, 1)  # importance scoring
#         self.attn = nn.MultiheadAttention(
#             hidden_size, num_heads=4, batch_first=True
#         )  # focused MHA

#         if mandatory_label_ids:
#             mand = torch.tensor(mandatory_label_ids, dtype=torch.long)
#             self.register_buffer("mandatory_ids", mand, persistent=False)
#         else:
#             self.mandatory_ids = None

#     def _build_gate_mask(self, entity_pred_ids: torch.Tensor) -> torch.Tensor:
#         """Boolean mask [B, L] where True marks mandatory tokens."""
#         if self.mandatory_ids is None or self.mandatory_ids.numel() == 0:
#             return torch.zeros_like(entity_pred_ids, dtype=torch.bool)
#         gate = torch.zeros_like(entity_pred_ids, dtype=torch.bool)
#         for lid in self.mandatory_ids:
#             gate |= (entity_pred_ids == lid)
#         return gate

#     def forward(self, input_ids, attention_mask, entity_pred_ids):
#         """
#         Args:
#             input_ids: [B, L]
#             attention_mask: [B, L] (1 real, 0 pad)
#             entity_pred_ids: [B, L] (token-level labels from Stage 1)
#         Returns:
#             h_updated: [B, L, H]
#             bias_mask: [B, L] (float >0 at entity tokens)
#             entity_pred_ids: [B, L] passthrough
#         """
#         B, L = input_ids.size()

#         # ---- contextual encoding + entity injection ----
#         enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         h = enc_out.last_hidden_state  # [B, L, H]
#         ent_emb = self.entity_embed(entity_pred_ids)  # [B, L, H]
#         h = h + ent_emb

#         # ---- importance scoring ----
#         scores = self.score_proj(h).squeeze(-1)  # [B, L]
#         scores = scores.masked_fill(attention_mask == 0, float("-inf"))
#         weights = torch.softmax(scores, dim=-1)  # [B, L]

#         # ---- hard gating (mandatory inclusion) ----
#         gate_mask = self._build_gate_mask(entity_pred_ids)  # [B, L] bool
#         K = max(1, int(self.top_k_ratio * L))
#         if self.max_k is not None:
#             K = min(K, int(self.max_k))

#         # union of mandatory ∪ topK rest
#         selected_idx = []
#         for b in range(B):
#             g_idx = torch.nonzero(gate_mask[b], as_tuple=False).squeeze(-1)  # [G] or []
#             G = g_idx.numel()

#             mask_rest = (~gate_mask[b]) & (attention_mask[b] == 1)
#             rest_scores = scores[b].clone()
#             rest_scores[~mask_rest] = float("-inf")
#             need = max(0, K - G)
#             if need > 0:
#                 k_take = min(need, int(mask_rest.sum().item()))
#                 topk = (
#                     torch.topk(rest_scores, k=k_take, dim=0).indices
#                     if k_take > 0
#                     else g_idx.new_empty(0)
#                 )
#                 sel = torch.cat([g_idx, topk], dim=0)
#             else:
#                 sel = g_idx if G > 0 else torch.topk(weights[b], k=K, dim=0).indices

#             sel = torch.unique(sel, sorted=True)
#             selected_idx.append(sel)

#         # pad selection to same K' across batch
#         Kp = max(s.numel() for s in selected_idx)
#         Kp = max(1, Kp)  # guard
#         idx_mat = torch.full(
#             (B, Kp), fill_value=0, dtype=torch.long, device=input_ids.device
#         )
#         valid_mask = torch.zeros((B, Kp), dtype=torch.bool, device=input_ids.device)
#         for b, s in enumerate(selected_idx):
#             k = s.numel()
#             if k > 0:
#                 idx_mat[b, :k] = s
#                 valid_mask[b, :k] = True

#         # ---- gather selected tokens ----
#         batch_arange = torch.arange(B, device=input_ids.device).unsqueeze(-1)
#         selected_tokens = h[batch_arange, idx_mat]  # [B, K', H]

#         # =========================================================
#         # FIXED PATCH: build additive attention mask for MHA
#         # - Positive bias for mandatory keys
#         # - Large negative for padded keys (fold key padding here)
#         # - Expand attn_mask to (B * num_heads, K', K') as required
#         # =========================================================
#         n_heads = getattr(self.attn, "num_heads", 4)

#         # bias on keys which are mandatory
#         key_bias = torch.zeros((B, Kp), dtype=selected_tokens.dtype, device=selected_tokens.device)  # [B, K']
#         selected_is_mand = gate_mask[batch_arange, idx_mat]  # [B, K'] bool
#         key_bias[selected_is_mand] = self.bias_strength

#         # additive bias [B, K', K'] (query-by-key)
#         attn_bias = key_bias.unsqueeze(1).expand(B, Kp, Kp).contiguous()  # [B, K', K']

#         # fold key padding into additive mask (avoid key_padding_mask warning)
#         neg_large = -1e4 if selected_tokens.dtype in (torch.float16, torch.bfloat16) else -1e9
#         # padded keys -> add big negative to logits so softmax ~ 0
#         pad_bias = (~valid_mask).unsqueeze(1).to(attn_bias.dtype) * neg_large  # [B, 1, K']
#         attn_bias = attn_bias + pad_bias  # [B, K', K']

#         # expand to (B * num_heads, K', K') as required by MHA for 3D attn_mask
#         attn_bias = attn_bias.repeat_interleave(n_heads, dim=0)  # [(B*n_heads), K', K']

#         # Focused multi-head attention with additive mask; no key_padding_mask
#         attn_out, _ = self.attn(
#             selected_tokens,               # [B, K', H]
#             selected_tokens,
#             selected_tokens,
#             attn_mask=attn_bias,           # [(B*n_heads), K', K']
#             need_weights=False,
#         )  # -> [B, K', H]
#         # =========================================================

#         # ---- write back to original sequence ----
#         h_updated = h.clone()
#         h_updated[batch_arange.expand(-1, Kp), idx_mat] = attn_out

#         # ---- bias mask (float >0 at entity tokens) ----
#         bias_mask = (entity_pred_ids > 0).float() * self.bias_strength  # [B, L]

#         return h_updated, bias_mask, entity_pred_ids


# # === Dataset Wrapper ===
# class DialogueDataset(Dataset):
#     def __init__(self, dialogues: List[str], tokenizer, max_length: int = 512):
#         self.dialogues = dialogues
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.dialogues)

#     def __getitem__(self, idx):
#         enc = self.tokenizer(
#             self.dialogues[idx],
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#         )
#         return {k: v.squeeze(0) for k, v in enc.items()}


# @torch.no_grad()
# def run_stage1_stage2_batch(dataloader, stage1_model, stage2_encoder, device):
#     all_embeds, all_biases, all_entity_ids = [], [], []
#     stage1_model.eval()
#     stage2_encoder.eval()
#     for batch in tqdm(dataloader, desc="Stage 1+2"):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         token_type_ids = batch.get("token_type_ids")
#         if token_type_ids is not None:
#             token_type_ids = token_type_ids.to(device)
#             ner_outputs = stage1_model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#             )
#         else:
#             ner_outputs = stage1_model(
#                 input_ids=input_ids, attention_mask=attention_mask
#             )

#         entity_pred_ids = torch.argmax(ner_outputs.logits, dim=-1)
#         contextual_embeds, bias_mask, _ = stage2_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             entity_pred_ids=entity_pred_ids,
#         )
#         all_embeds.append(contextual_embeds.cpu())
#         all_biases.append(bias_mask.cpu())
#         all_entity_ids.append(entity_pred_ids.cpu())
#     return (
#         torch.cat(all_embeds, dim=0),
#         torch.cat(all_biases, dim=0),
#         torch.cat(all_entity_ids, dim=0),
#     )


# def preprocess_and_save(
#     split_name,
#     csv_path,
#     outdir,
#     stage1_model,
#     stage1_tokenizer,
#     stage2_encoder,
#     device,
#     batch_size=1,
# ):
#     df = pd.read_csv(csv_path)
#     dialogues = df["dialogue"].astype(str).tolist()
#     dataset = DialogueDataset(dialogues, stage1_tokenizer)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     print(f"Processing {split_name} set: {len(dialogues)} samples...")
#     embeds, bias_masks, entity_ids = run_stage1_stage2_batch(
#         dataloader, stage1_model, stage2_encoder, device
#     )
#     os.makedirs(outdir, exist_ok=True)
#     save_path = os.path.join(outdir, f"{split_name}_stage2_outputs.pt")
#     torch.save(
#         {
#             "contextual_embeds": embeds,
#             "bias_masks": bias_masks,
#             "entity_pred_ids": entity_ids,
#             "dialogues": dialogues,
#             "meta": {"split": split_name},
#         },
#         save_path,
#     )
#     print(f"Saved: {save_path}")


# def _load_mandatory_ids(stage1_dir: str, stage1_model) -> List[int]:
#     """
#     Try (1) model.config.mandatory_label_ids, then (2) sidecar JSON, else (3) derive from id2label.
#     """
#     # (1) from config
#     ids = getattr(stage1_model.config, "mandatory_label_ids", None)
#     if isinstance(ids, (list, tuple)) and len(ids) > 0:
#         print(f"[DSDA] Using mandatory ids from config: {ids}")
#         return [int(x) for x in ids]

#     # (2) sidecar JSON
#     sidecar = os.path.join(stage1_dir, "mandatory_labels.json")
#     if os.path.isfile(sidecar):
#         try:
#             with open(sidecar, "r") as f:
#                 data = json.load(f)
#             ids = data.get("mandatory_label_ids", [])
#             if ids:
#                 print(f"[DSDA] Using mandatory ids from sidecar: {ids}")
#                 return [int(x) for x in ids]
#         except Exception as e:
#             print(f"[DSDA] Warning: could not read sidecar {sidecar}: {e}")

#     # (3) infer from id2label
#     id2label = getattr(stage1_model.config, "id2label", None)
#     if isinstance(id2label, dict):
#         try:
#             id2label = {int(k): v for k, v in id2label.items()}
#         except Exception:
#             pass
#         want_types = {"DISEASE", "CHEMICAL"}
#         keep = []
#         for i, name in id2label.items():
#             if isinstance(name, str) and "-" in name:
#                 _, et = name.split("-", 1)
#                 if et.upper() in want_types:
#                     keep.append(int(i))
#         if keep:
#             print(f"[DSDA] Inferred mandatory ids from id2label: {keep}")
#             return keep

#     print("[DSDA] No mandatory ids found; gating disabled.")
#     return []


# if __name__ == "__main__":
#     # Paths
#     stage1_dir = "/home/raza/EMRV1/EMRV2/outputs/stage1.2_outputs/final_model"
#     outdir = "/home/raza/EMRV1/EMRV2/outputs/stage2.2_outputs"
#     datadir = "/home/raza/EMRV1/EMRV2/datasets"

#     splits = {
#         "train": os.path.join(datadir, "ACI-TaskB-TrainingSet.csv"),
#         "val": os.path.join(datadir, "ACI-TaskB-ValidationSet.csv"),
#         "test": os.path.join(datadir, "ACI-TaskB-TestingSet.csv"),
#     }

#     print(f"Loading Stage 1 model from {stage1_dir} ...")
#     stage1_model = AutoModelForTokenClassification.from_pretrained(stage1_dir)
#     stage1_tokenizer = AutoTokenizer.from_pretrained(stage1_dir)

#     backbone = "emilyalsentzer/Bio_ClinicalBERT"
#     hidden_size = stage1_model.config.hidden_size
#     entity_label_count = stage1_model.config.num_labels
#     mandatory_ids = _load_mandatory_ids(stage1_dir, stage1_model)

#     stage2_encoder = BiasedAttentionEncoder(
#         backbone_name=backbone,
#         hidden_size=hidden_size,
#         entity_label_count=entity_label_count,
#         bias_strength=3.0,
#         top_k_ratio=0.25,
#         mandatory_label_ids=mandatory_ids,
#         max_k=256,  # optional cap
#     )

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     stage1_model = stage1_model.to(device)
#     stage2_encoder = stage2_encoder.to(device)

#     for split_name, csv_path in splits.items():
#         preprocess_and_save(
#             split_name,
#             csv_path,
#             outdir,
#             stage1_model,
#             stage1_tokenizer,
#             stage2_encoder,
#             device,
#             batch_size=1,
#         )

#     print("\nStage 2 complete — embeddings & bias masks saved.\n")

