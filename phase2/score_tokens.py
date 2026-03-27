#!/usr/bin/env python3
"""Phase 2: semi-offline token scoring.

Loads D_train, runs target model and frozen reference model in dual forward,
computes per-speech-token excess loss, generates top-k% mask, saves to disk.

Usage:
    python -m phase2.score_tokens \
        --train-data-list phase1_outputs/data/train.data.list \
        --target-ckpt phase1_outputs/rm/rm_frozen.pt \
        --ref-ckpt phase1_outputs/rm/rm_frozen.pt \
        --config phase1_outputs/phase1_cosyvoice2.yaml \
        --qwen-pretrain-path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
        --onnx-path pretrained_models/CosyVoice2-0.5B \
        --output-dir phase2_outputs \
        --topk-ratio 0.6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(config_path, qwen_pretrain_path, checkpoint_path, onnx_path, device):
    """Load a Qwen2LM model from config + checkpoint."""
    os.environ['onnx_path'] = onnx_path
    from hyperpyyaml import load_hyperpyyaml

    override_dict = {
        'flow': None,
        'hift': None,
        'hifigan': None,
        'qwen_pretrain_path': qwen_pretrain_path,
    }
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)

    model = configs['llm']

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Remove 'step'/'epoch' keys that aren't model params
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(('step', 'epoch')) and isinstance(v, torch.Tensor)}
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        logger.warning("No checkpoint loaded (path: %s)", checkpoint_path)

    model = model.to(device)
    model.eval()
    return model


def build_dataloader(data_list_file, config_path, qwen_pretrain_path, onnx_path,
                     num_workers=0, prefetch=100):
    """Build CosyVoice2 dataset + dataloader for scoring (no shuffle)."""
    os.environ['onnx_path'] = onnx_path
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.dataset.dataset import Dataset
    from torch.utils.data import DataLoader

    override_dict = {
        'flow': None,
        'hift': None,
        'hifigan': None,
        'qwen_pretrain_path': qwen_pretrain_path,
    }
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)

    data_pipeline = configs['data_pipeline']
    dataset = Dataset(data_list_file, data_pipeline=data_pipeline,
                      mode='dev', gan=False, dpo=False,
                      shuffle=False, partition=False)
    dataloader = DataLoader(dataset, batch_size=None,
                            pin_memory=False,
                            num_workers=num_workers,
                            prefetch_factor=prefetch if num_workers > 0 else None)
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Phase 2: semi-offline token scoring")
    parser.add_argument("--train-data-list", required=True,
                        help="data.list file for D_train (parquet paths)")
    parser.add_argument("--target-ckpt", required=True,
                        help="Target model checkpoint (.pt)")
    parser.add_argument("--ref-ckpt", required=True,
                        help="Frozen reference model checkpoint (.pt)")
    parser.add_argument("--config", required=True,
                        help="CosyVoice2 YAML config (e.g. phase1_cosyvoice2.yaml)")
    parser.add_argument("--qwen-pretrain-path", required=True,
                        help="Path to CosyVoice-BlankEN")
    parser.add_argument("--onnx-path", required=True,
                        help="Path to dir containing campplus.onnx etc.")
    parser.add_argument("--output-dir", default="phase2_outputs",
                        help="Output directory for scores")
    parser.add_argument("--topk-ratio", type=float, default=0.6,
                        help="Top-k%% ratio for token selection")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Setup PYTHONPATH for CosyVoice imports ──
    cosyvoice_root = str(Path(__file__).resolve().parents[1] / ".." / "CosyVoice")
    # Try to find CosyVoice root from common locations
    for candidate in [
        "/data/zhenghao/repos/CosyVoice",
        str(Path(__file__).resolve().parents[1] / "CosyVoice"),
    ]:
        if Path(candidate).exists():
            cosyvoice_root = candidate
            break
    for p in [cosyvoice_root, os.path.join(cosyvoice_root, "third_party", "Matcha-TTS")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    logger.info("Loading target model from %s", args.target_ckpt)
    target_model = load_model(
        args.config, args.qwen_pretrain_path, args.target_ckpt, args.onnx_path, device
    )

    logger.info("Loading reference model from %s", args.ref_ckpt)
    ref_model = load_model(
        args.config, args.qwen_pretrain_path, args.ref_ckpt, args.onnx_path, device
    )
    # Freeze reference model explicitly
    for param in ref_model.parameters():
        param.requires_grad = False

    logger.info("Building dataloader from %s", args.train_data_list)
    dataloader = build_dataloader(
        args.train_data_list, args.config, args.qwen_pretrain_path, args.onnx_path,
        num_workers=args.num_workers, prefetch=args.prefetch,
    )

    # ── Import scoring functions ──
    from phase2.scoring import (
        extract_speech_tokens, compute_speech_token_losses,
        compute_excess_loss, build_topk_mask,
    )

    # ── Run scoring ──
    all_results = []
    batch_idx = 0
    total_tokens = 0
    selected_tokens = 0
    t0 = time.time()

    logger.info("Starting scoring with topk_ratio=%.2f", args.topk_ratio)

    for batch in dataloader:
        utts = batch.get('utts', [])
        if not utts:
            continue

        # Extract speech tokens once, shared by both models
        speech_token, speech_token_len = extract_speech_tokens(target_model, batch, device)

        # Dual forward with shared speech tokens
        target_losses, _ = compute_speech_token_losses(target_model, batch, device, speech_token, speech_token_len)
        ref_losses, _ = compute_speech_token_losses(ref_model, batch, device, speech_token, speech_token_len)

        # Excess loss
        deltas = compute_excess_loss(target_losses, ref_losses)

        # Top-k mask
        masks = build_topk_mask(deltas, topk_ratio=args.topk_ratio)

        # Collect results
        for i, utt in enumerate(utts):
            n_tokens = target_losses[i].numel()
            n_selected = masks[i].sum().item()
            total_tokens += n_tokens
            selected_tokens += n_selected

            all_results.append({
                'utt': utt,
                'speech_token_score': deltas[i],        # 1-D float tensor
                'speech_token_mask': masks[i],           # 1-D bool tensor
                'target_loss': target_losses[i],         # 1-D float tensor
                'ref_loss': ref_losses[i],               # 1-D float tensor
            })

        batch_idx += 1
        if batch_idx % 5 == 0:
            logger.info("Scored %d batches, %d samples so far", batch_idx, len(all_results))

    elapsed = time.time() - t0
    logger.info("Scoring done: %d samples, %d batches in %.1fs", len(all_results), batch_idx, elapsed)

    if total_tokens > 0:
        actual_ratio = selected_tokens / total_tokens
        logger.info("Total speech tokens: %d, selected: %d (%.1f%%)",
                     total_tokens, selected_tokens, actual_ratio * 100)
    else:
        actual_ratio = 0.0
        logger.warning("No speech tokens found!")

    # ── Save results ──
    # Save per-sample scores as a single .pt file
    scores_path = output_dir / "token_scores.pt"
    save_data = {
        'results': all_results,
        'topk_ratio': args.topk_ratio,
        'target_ckpt': args.target_ckpt,
        'ref_ckpt': args.ref_ckpt,
        'total_tokens': total_tokens,
        'selected_tokens': selected_tokens,
        'actual_ratio': actual_ratio,
    }
    torch.save(save_data, scores_path)
    logger.info("Saved scores to %s", scores_path)

    # Save summary
    summary = {
        'num_samples': len(all_results),
        'num_batches': batch_idx,
        'total_speech_tokens': total_tokens,
        'selected_tokens': selected_tokens,
        'topk_ratio': args.topk_ratio,
        'actual_selection_ratio': actual_ratio,
        'target_ckpt': args.target_ckpt,
        'ref_ckpt': args.ref_ckpt,
        'elapsed_seconds': elapsed,
        'output_file': str(scores_path),
    }
    summary_path = output_dir / "phase2_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
