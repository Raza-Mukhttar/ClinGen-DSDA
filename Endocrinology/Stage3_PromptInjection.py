

# Stage3_PromptInjection.py
# Prompt Projection (Gated + Biased, DSDA-aligned)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class PromptProjection(nn.Module):
    """
    Stage 3: Prompt Projection Module
    Converts [B, L, H] contextual embeddings into [B, P, H] prompt vectors,
    using entity-aware gating (hard) and biasing (soft).

    Inputs (forward):
      - contextual_embeds: [B, L, H]  (from Stage 2)
      - entity_mask_or_bias: [B, L]   (float bias_mask from Stage 2) OR [B, L] long ids
      - attention_mask: Optional [B, L] (1=real, 0=pad), used to suppress padding in pooling
      - entity_pred_ids: Optional [B, L] long (alternative to entity_mask_or_bias if you prefer)
      - mandatory_label_ids: Optional list[int]; tokens with these ids are *hard-gated* in

    Modes:
      - 'attentive' (recommended): linear->tanh->linear over tokens + additive bias then softmax over L
      - 'mean': masked mean pooling (with fallback)
      - 'ffn': pooled feature reduced to prompt_len*H via MLP

    Notes:
      - If provided a float mask (bias_mask), >0 is considered "entity" (and used as additive bias).
      - If provided label ids, >0 is entity, plus mandatory_label_ids are *forced on* (hard gate=1).
      - If the (entity or valid) mask is all-zero for a sample, we fallback to include all non-pad tokens
        (or all tokens if attention_mask is None).
    """

    def __init__(
        self,
        hidden_size: int,
        prompt_len: int = 16,
        mode: str = "attentive",
        attentive_hidden: int = 128,
        mask_bias_strength: float = 2.0,   # additive bias magnitude for entity tokens
        debug: bool = False,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.mode = mode
        self.hidden_size = hidden_size
        self.mask_bias_strength = float(mask_bias_strength)
        self.debug = debug

        if self.mode == "attentive":
            self.entity_att = nn.Sequential(
                nn.Linear(hidden_size, attentive_hidden),
                nn.Tanh(),
                nn.Linear(attentive_hidden, prompt_len),
            )
        elif self.mode == "ffn":
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, prompt_len * hidden_size),
            )
        elif self.mode == "mean":
            pass
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    @staticmethod
    def _to_entity_mask(
        entity_mask_or_bias: torch.Tensor,
        entity_pred_ids: Optional[torch.Tensor],
        mandatory_label_ids: Optional[List[int]],
    ) -> torch.Tensor:
        """
        Build a binary entity mask [B, L] where 1 marks entity tokens.
        - If entity_mask_or_bias is float: treat >0 as entity.
        - Else if long: treat >0 as entity.
        - If entity_pred_ids provided, OR it in as (>0).
        - If mandatory_label_ids provided, OR those classes in as hard gate.
        """
        if entity_mask_or_bias is None and entity_pred_ids is None:
            raise ValueError("Provide at least `entity_mask_or_bias` or `entity_pred_ids`.")

        # Start with zeros on the appropriate device/dtype
        if entity_mask_or_bias is not None:
            B, L = entity_mask_or_bias.shape
            device = entity_mask_or_bias.device
        else:
            B, L = entity_pred_ids.shape
            device = entity_pred_ids.device

        entity_mask = torch.zeros((B, L), dtype=torch.float32, device=device)

        # Case 1: got a float bias/mask directly
        if entity_mask_or_bias is not None and entity_mask_or_bias.dtype.is_floating_point:
            entity_mask = (entity_mask_or_bias > 0).float()
        # Case 2: got label ids in entity_mask_or_bias
        elif entity_mask_or_bias is not None and not entity_mask_or_bias.dtype.is_floating_point:
            entity_mask = (entity_mask_or_bias > 0).float()

        # OR in tokens from explicit entity_pred_ids if supplied
        if entity_pred_ids is not None:
            entity_mask = torch.logical_or(entity_mask.bool(), (entity_pred_ids > 0)).float()

        # Hard-gate mandatory label ids, if provided
        if mandatory_label_ids:
            mand = torch.as_tensor(mandatory_label_ids, dtype=torch.long, device=device)
            # build OR across all mandatory ids
            if entity_pred_ids is None:
                # if we only had a float mask, we cannot detect *which* class; leave as-is
                pass
            else:
                gate = torch.zeros((B, L), dtype=torch.bool, device=device)
                for lid in mand:
                    gate |= (entity_pred_ids == lid)
                entity_mask = torch.logical_or(entity_mask.bool(), gate).float()

        return entity_mask  # [B, L] float in {0,1}

    @staticmethod
    def _fallback_mask(entity_mask: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        If entity_mask is all-zero for a sample, fallback to include all non-pad tokens (or all tokens).
        """
        B, L = entity_mask.shape
        out = entity_mask.clone()
        zero_rows = (out.sum(dim=1) == 0)  # [B]
        if zero_rows.any():
            if attention_mask is not None:
                out[zero_rows] = attention_mask[zero_rows].float()
            else:
                out[zero_rows] = 1.0
        return out

    def forward(
        self,
        contextual_embeds: torch.Tensor,              # [B, L, H]
        entity_mask_or_bias: Optional[torch.Tensor],  # [B, L] float mask/bias OR long ids
        attention_mask: Optional[torch.Tensor] = None,# [B, L] (1=real, 0=pad), optional but recommended
        *,
        entity_pred_ids: Optional[torch.Tensor] = None,         # [B, L] long ids
        mandatory_label_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B, L, H = contextual_embeds.shape
        assert H == self.hidden_size, f"H mismatch: got {H}, expected {self.hidden_size}"

        # ---- build binary entity mask (gated) ----
        entity_mask = self._to_entity_mask(entity_mask_or_bias, entity_pred_ids, mandatory_label_ids)  # [B, L]
        # suppress pads if attention_mask provided
        if attention_mask is not None:
            entity_mask = entity_mask * attention_mask.float()
        # fallback if all-zero
        entity_mask = self._fallback_mask(entity_mask, attention_mask)

        if self.mode == "mean":
            # weighted mean over tokens marked by entity_mask
            weights = entity_mask.unsqueeze(-1)                        # [B, L, 1]
            masked = contextual_embeds * weights                       # [B, L, H]
            summed = masked.sum(dim=1)                                 # [B, H]
            denom = weights.sum(dim=1).clamp(min=1.0)                  # [B, 1]
            pooled = summed / denom                                    # [B, H]
            prompt_embeds = pooled.unsqueeze(1).expand(B, self.prompt_len, H)  # [B, P, H]

        elif self.mode == "ffn":
            weights = entity_mask.unsqueeze(-1)
            masked = contextual_embeds * weights
            pooled = masked.sum(dim=1) / (weights.sum(dim=1).clamp(min=1.0))   # [B, H]
            ffn_out = self.ffn(pooled)                                          # [B, P*H]
            prompt_embeds = ffn_out.view(B, self.prompt_len, H)                 # [B, P, H]

        elif self.mode == "attentive":
            # base scores over tokens for each prompt slot
            attn_scores = self.entity_att(contextual_embeds)  # [B, L, P]

            # build additive bias:
            #   +mask_bias_strength for entity tokens, -inf (or very large negative) for pads (if any)
            add_bias = entity_mask.unsqueeze(-1) * self.mask_bias_strength  # [B, L, P]

            if attention_mask is not None:
                # mask out pads: add a large negative so softmax ~0
                neg_large = -1e4 if contextual_embeds.dtype in (torch.float16, torch.bfloat16) else -1e9
                add_bias = add_bias + (1.0 - attention_mask.float()).unsqueeze(-1) * neg_large

            # final scores then softmax over token dimension
            attn_scores = attn_scores + add_bias
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, L, P]

            # weighted sum over tokens -> prompts
            # (blh, blp -> bph)
            prompt_embeds = torch.einsum("blh,blp->bph", contextual_embeds, attn_weights)

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.debug:
            with torch.no_grad():
                print("=" * 50)
                print("Stage 3 Prompt Embeddings Output:")
                print(f"Shape: {tuple(prompt_embeds.shape)}")
                m = prompt_embeds
                print(
                    f"Mean: {m.mean().item():.4f}, Std: {m.std().item():.4f}, "
                    f"Min: {m.min().item():.4f}, Max: {m.max().item():.4f}"
                )
                print("First prompt vec (batch 0, first 5 dims):",
                      m[0, 0, :5].detach().cpu().tolist() if B > 0 else "N/A")
                print("=" * 50)

        return prompt_embeds  # [B, P, H]
