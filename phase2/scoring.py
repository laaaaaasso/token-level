"""Phase 2 core: per-speech-token loss extraction and excess loss computation.

Given a CosyVoice2 Qwen2LM model and a batch, computes per-speech-token CE loss.
Then computes excess loss L_delta = L_target - L_ref and generates top-k% mask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

IGNORE_ID = -1


def extract_speech_tokens(model, batch, device):
    """Extract speech tokens from batch (online if needed). Done once, shared by both models.

    Returns:
        speech_token: [B, T_speech] padded
        speech_token_len: [B]
    """
    if 'speech_token' not in batch:
        speech_token, speech_token_len = model.speech_token_extractor.inference(
            batch['whisper_feat'], batch['whisper_feat_len'], device
        )
    else:
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
    return speech_token, speech_token_len


def _build_unistream_lm_inputs(model, batch, device, speech_token, speech_token_len):
    """Build unistream LM input/target from batch + pre-extracted speech tokens.

    Returns:
        lm_target: [B, T] with IGNORE_ID at non-speech positions
        lm_input:  [B, T, D] embeddings
        lm_input_len: [B]
    """
    text_token = batch['text_token'].to(device)
    text_token_len = batch['text_token_len'].to(device)

    text_token_emb = model.llm.model.model.embed_tokens(text_token)
    speech_token_emb = model.speech_embedding(speech_token)

    sos_emb = model.llm_embedding.weight[model.sos].reshape(1, 1, -1)
    task_id_emb = model.llm_embedding.weight[model.task_id].reshape(1, 1, -1)

    text_token_emb_list = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
    speech_token_list = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
    speech_token_emb_list = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)

    lm_targets = []
    lm_inputs = []
    for i in range(len(text_token_emb_list)):
        t = torch.tensor(
            [IGNORE_ID] * (1 + int(text_token_len[i].item()))
            + speech_token_list[i].tolist()
            + [model.eos_token]
        )
        inp = torch.cat([
            sos_emb.squeeze(0),
            text_token_emb_list[i],
            task_id_emb.squeeze(0),
            speech_token_emb_list[i],
        ], dim=0)
        lm_targets.append(t)
        lm_inputs.append(inp)

    lm_input_len = torch.tensor([x.size(0) for x in lm_inputs], dtype=torch.int32)
    lm_input = pad_sequence(lm_inputs, batch_first=True, padding_value=IGNORE_ID)
    lm_target = pad_sequence(lm_targets, batch_first=True, padding_value=IGNORE_ID).to(device)

    return lm_target, lm_input, lm_input_len


@torch.inference_mode()
def compute_speech_token_losses(model, batch, device, speech_token, speech_token_len):
    """Run model forward and return per-speech-token CE loss.

    Args:
        model: Qwen2LM (unwrapped, on device, eval mode)
        batch: dict from CosyVoice2 data pipeline
        device: torch device
        speech_token: [B, T_speech] pre-extracted (shared between target/ref)
        speech_token_len: [B]

    Returns:
        per_token_loss: list of 1-D tensors, one per sample
        utts: list of sample IDs
    """
    lm_target, lm_input, lm_input_len = \
        _build_unistream_lm_inputs(model, batch, device, speech_token, speech_token_len)

    # Forward through LLM
    lm_output, _ = model.llm(lm_input, lm_input_len.to(device))
    logits = model.llm_decoder(lm_output)  # [B, T, V]

    # Per-token CE loss (no reduction)
    B, T, V = logits.shape
    flat_logits = logits.view(-1, V)
    flat_target = lm_target.view(-1)
    valid_mask = (flat_target != IGNORE_ID)
    safe_target = flat_target.clone()
    safe_target[~valid_mask] = 0
    flat_loss = F.cross_entropy(flat_logits, safe_target, reduction='none')
    flat_loss[~valid_mask] = 0.0
    token_loss = flat_loss.view(B, T)

    # Extract only speech token positions for each sample
    speech_mask = (lm_target != IGNORE_ID)  # [B, T]
    results = []
    for i in range(B):
        losses_i = token_loss[i][speech_mask[i]]
        results.append(losses_i.cpu())

    utts = batch.get('utts', [f'sample_{i}' for i in range(B)])
    return results, utts


def compute_excess_loss(target_losses, ref_losses):
    """Compute L_delta = L_target - L_ref for each sample."""
    deltas = []
    for t_loss, r_loss in zip(target_losses, ref_losses):
        assert t_loss.shape == r_loss.shape, \
            f"Shape mismatch: target {t_loss.shape} vs ref {r_loss.shape}"
        deltas.append(t_loss - r_loss)
    return deltas


def build_topk_mask(deltas, topk_ratio=0.6):
    """Build top-k% token mask based on excess loss (higher = more important).

    Selects tokens with the highest L_delta (most room for improvement).
    """
    masks = []
    for delta in deltas:
        n = delta.numel()
        if n == 0:
            masks.append(torch.zeros(0, dtype=torch.bool))
            continue
        k = max(1, int(round(n * topk_ratio)))
        _, topk_idx = torch.topk(delta, k)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[topk_idx] = True
        masks.append(mask)
    return masks
