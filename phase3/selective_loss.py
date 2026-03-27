"""Selective SLM loss: only top-k% speech tokens contribute to training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_ID = -1


def compute_selective_slm_loss(logits, lm_target, speech_masks):
    """Compute CE loss only on selected speech token positions.

    Args:
        logits: [B, T, V] model output logits
        lm_target: [B, T] target ids, IGNORE_ID at non-speech positions
        speech_masks: list of B elements. Each element is either:
            - a bool tensor [T_speech_i] where True = selected
            - None (meaning use all speech tokens for this sample)

    Returns:
        loss: scalar tensor with grad
        n_selected: number of tokens that contributed to loss
    """
    B, T, V = logits.shape
    device = logits.device

    # Per-token CE loss (no reduction)
    flat_logits = logits.view(-1, V)
    flat_target = lm_target.view(-1)
    valid = (flat_target != IGNORE_ID)
    safe_target = flat_target.clone()
    safe_target[~valid] = 0
    flat_loss = F.cross_entropy(flat_logits, safe_target, reduction='none')
    token_loss = flat_loss.view(B, T)
    # Zero out invalid positions (padding / text / sos / task_id)
    token_loss = token_loss * valid.view(B, T).float()

    # Build selection mask aligned to full sequence
    selection = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        # Speech token positions in the full sequence
        pos_i = (lm_target[i] != IGNORE_ID).nonzero(as_tuple=True)[0]
        mask_i = speech_masks[i] if speech_masks is not None and i < len(speech_masks) else None

        if mask_i is None:
            # No mask: use all speech tokens (fallback for CV or missing utts)
            selection[i, pos_i] = True
        else:
            n = min(len(pos_i), len(mask_i))
            if n > 0:
                mask_i_dev = mask_i[:n].to(device)
                selection[i, pos_i[:n]] = mask_i_dev

    n_selected = selection.sum().item()
    if n_selected == 0:
        # Avoid NaN: return zero loss that still has grad
        return (token_loss * 0.0).sum(), 0

    loss = (token_loss * selection.float()).sum() / n_selected
    return loss, int(n_selected)
