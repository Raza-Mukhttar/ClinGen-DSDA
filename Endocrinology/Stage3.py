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







