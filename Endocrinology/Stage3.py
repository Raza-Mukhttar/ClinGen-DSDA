import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptProjection(nn.Module):
    """
    Stage 3: Prompt Projection Module
    Converts [B, L, H] contextual embeddings and [B, L] entity mask into [B, P, H] prompt vectors.
    - Supports: attentive pooling (recommended), mean pooling, or FFN reduction.
    """
    def __init__(self, hidden_size, prompt_len=16, mode='attentive', attentive_hidden=128):
        super().__init__()
        self.prompt_len = prompt_len
        self.mode = mode
        self.hidden_size = hidden_size

        if self.mode == 'attentive':
            self.entity_att = nn.Sequential(
                nn.Linear(hidden_size, attentive_hidden),
                nn.Tanh(),
                nn.Linear(attentive_hidden, prompt_len)
            )

        elif self.mode == 'ffn':
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, prompt_len * hidden_size)
            )

    def forward(self, contextual_embeds, entity_mask):
        """
        Args:
            contextual_embeds: [B, L, H] from Stage 2
            entity_mask: [B, L] float tensor (1 = entity, 0 = non-entity)
        Returns:
            prompt_embeds: [B, P, H] (prompt_len = P)
        """
        B, L, H = contextual_embeds.shape

        if self.mode == 'mean':
            mask = entity_mask.unsqueeze(-1)
            masked = contextual_embeds * mask
            summed = masked.sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            pooled = summed / count
            prompt_embeds = pooled.unsqueeze(1).expand(B, self.prompt_len, H)

        elif self.mode == 'ffn':
            mask = entity_mask.unsqueeze(-1)
            masked = contextual_embeds * mask
            pooled = masked.sum(dim=1) / (mask.sum(dim=1).clamp(min=1))
            ffn_out = self.ffn(pooled)
            prompt_embeds = ffn_out.view(B, self.prompt_len, H)

        elif self.mode == 'attentive':
            mask = entity_mask  # [B, L]
            if (mask.sum(dim=1) == 0).any():
                print("Warning: All-zero entity_mask detected, using fallback (all tokens).")
                mask[mask.sum(dim=1) == 0, :] = 1
            attn_scores = self.entity_att(contextual_embeds)  # [B, L, prompt_len]
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, L, prompt_len]
            prompt_embeds = torch.einsum('blh,blp->bph', contextual_embeds, attn_weights)

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # PRINT PROMPT OUTPUT (for debug)
        print("=" * 50)
        print("Stage 3 Prompt Embeddings Output:")
        print(f"Shape: {prompt_embeds.shape}")
        # Print summary statistics
        print(f"Mean: {prompt_embeds.mean().item():.4f}, Std: {prompt_embeds.std().item():.4f}, Min: {prompt_embeds.min().item():.4f}, Max: {prompt_embeds.max().item():.4f}")
        print("First prompt vector (batch 0):", prompt_embeds[0, 0, :5].detach().cpu().numpy(), "...")  # Print first 5 dims for clarity
        print("=" * 50)
        return prompt_embeds  # [B, P, H]







# # file: stage3_prompt_projection.py

# file: stage3_prompt_projection_v2.py

# import os
# import torch
# import torch.nn as nn
# from tqdm import tqdm

# # ===================== SETTINGS =====================
# PROJECTION_MODE = "attentive"   # "mean" | "ffn" | "attentive"
# PROMPT_COUNT = 16                # number of prompt vectors per dialogue
# USE_FLOAT16 = True              # compress to fp16 if embeddings are large
# BATCH_SIZE = 32                 # batched inference for speed
# # ====================================================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ---------------- PATHS ----------------
# STAGE2_OUT = "/home/raza/EMRV1/EMRV2/outputs/stage222_outputs/stage222_outputs.pt"
# OUTDIR = "/home/raza/EMRV1/EMRV2/outputs/stage333_outputs"
# os.makedirs(OUTDIR, exist_ok=True)

# # ---------------- MODEL ----------------
# class PromptProjection(nn.Module):
#     """
#     Converts contextual embeddings [B,L,H] + bias_mask [B,L]
#     → prompt embeddings [B,P,H].
#     Supports mean, ffn, and attentive modes.
#     """
#     def __init__(self, hidden_size, prompt_count=5, projection_mode="attentive"):
#         super().__init__()
#         self.mode = projection_mode
#         self.prompt_count = prompt_count
#         self.hidden_size = hidden_size

#         if self.mode == "ffn":
#             self.ffn = nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(hidden_size, hidden_size),
#             )
#         elif self.mode == "attentive":
#             self.att_proj = nn.Linear(hidden_size, prompt_count)

#     def forward(self, contextual_embeds, bias_mask):
#         # Align dtype to module parameters
#         dtype = next(self.parameters()).dtype if any(p.requires_grad for p in self.parameters()) else contextual_embeds.dtype
#         contextual_embeds = contextual_embeds.to(dtype)
#         bias_mask = bias_mask.to(contextual_embeds.dtype)

#         mask = bias_mask.clone().bool()
#         if (mask.sum(dim=1) == 0).any():
#             mask[mask.sum(dim=1) == 0, :] = True  # prevent div/softmax NaN

#         if self.mode == "mean":
#             masked = contextual_embeds * mask.unsqueeze(-1)
#             denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
#             pooled = masked.sum(dim=1) / denom
#             prompt_embeds = pooled.unsqueeze(1).repeat(1, self.prompt_count, 1)

#         elif self.mode == "ffn":
#             masked = contextual_embeds * mask.unsqueeze(-1)
#             pooled = masked.mean(dim=1)
#             proj = self.ffn(pooled)
#             prompt_embeds = proj.unsqueeze(1).repeat(1, self.prompt_count, 1)

#         elif self.mode == "attentive":
#             scores = self.att_proj(contextual_embeds)  # [B,L,P]
#             scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
#             attn = torch.softmax(scores, dim=1)
#             prompt_embeds = torch.einsum("blh,blp->bph", contextual_embeds, attn)
#         else:
#             raise ValueError(f"Invalid projection mode: {self.mode}")
#         return prompt_embeds


# # ---------------- RUN STAGE 3 ----------------
# @torch.no_grad()
# def run_stage3():
#     print(f"Loading Stage 2 outputs from {STAGE2_OUT}")
#     data = torch.load(STAGE2_OUT, map_location="cpu")
#     contextual_embeds, bias_masks = data["contextual_embeds"], data["bias_masks"]

#     if isinstance(contextual_embeds, list):
#         contextual_embeds = torch.cat(contextual_embeds, dim=0)
#         bias_masks = torch.cat(bias_masks, dim=0)

#     if USE_FLOAT16:
#         contextual_embeds = contextual_embeds.half()

#     B, L, H = contextual_embeds.shape
#     print(f"Loaded embeddings: {B}×{L}×{H}")

#     projector = PromptProjection(hidden_size=H,
#                                  prompt_count=PROMPT_COUNT,
#                                  projection_mode=PROJECTION_MODE).to(DEVICE)

#     if USE_FLOAT16:
#         projector = projector.half()

#     projector.eval()

#     prompt_batches = []
#     for i in tqdm(range(0, B, BATCH_SIZE), desc="Projecting batches"):
#         h = contextual_embeds[i:i+BATCH_SIZE].to(DEVICE)
#         m = bias_masks[i:i+BATCH_SIZE].to(DEVICE)
#         prompts = projector(h, m)
#         prompt_batches.append(prompts.cpu())

#     prompt_embeds = torch.cat(prompt_batches, dim=0)
#     out_path = os.path.join(OUTDIR, f"stage3_prompts_{PROJECTION_MODE}.pt")

#     torch.save({"prompt_embeds": prompt_embeds}, out_path)
#     print(f"✅ Stage 3 complete — saved {prompt_embeds.shape} prompt embeddings → {out_path}")


# if __name__ == "__main__":
#     run_stage3()
