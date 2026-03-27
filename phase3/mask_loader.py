"""Load Phase 2 token masks for use in Phase 3 selective training."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_token_masks(scores_path):
    """Load Phase 2 token_scores.pt and return {utt: mask} dict.

    Args:
        scores_path: path to token_scores.pt from Phase 2

    Returns:
        mask_dict: {utt_id: bool tensor of shape [T_speech+1]}
            The +1 is for the eos token, matching Phase 2's scoring convention.
    """
    data = torch.load(scores_path, map_location='cpu')
    mask_dict = {}
    for r in data['results']:
        mask_dict[r['utt']] = r['speech_token_mask']
    topk_ratio = data.get('topk_ratio', 0.6)
    logger.info("Loaded %d token masks (topk_ratio=%.2f) from %s",
                len(mask_dict), topk_ratio, scores_path)
    return mask_dict
