#!/usr/bin/env python3
"""Stage C2: Unified full-token evaluation of all model checkpoints.

Loads each model checkpoint and evaluates on the CV set using ALL speech tokens
(no selective masking), so that losses are directly comparable across all runs.

Usage:
    conda activate cosyvoice2
    python stage_c_analysis/unified_eval.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_ROOT = Path("/data/zhenghao/repos/CosyVoice")
for p in [str(COSYVOICE_ROOT), str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ['onnx_path'] = str(PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B")

CONFIG_PATH = PROJECT_ROOT / "phase3_outputs" / "selective" / "phase3_cosyvoice2.yaml"
CV_DATA = PROJECT_ROOT / "phase1_outputs" / "data" / "cv.data.list"
QWEN_PATH = PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B" / "CosyVoice-BlankEN"
ONNX_PATH = PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B"
OUTPUT_DIR = Path(__file__).resolve().parent

IGNORE_ID = -1

# All model checkpoints to evaluate
MODELS = {
    "selective":      PROJECT_ROOT / "phase3_outputs" / "selective"      / "checkpoints",
    "full_baseline":  PROJECT_ROOT / "phase3_outputs" / "full_baseline"  / "checkpoints",
    "random_mask":    PROJECT_ROOT / "phase3_outputs" / "random_mask"    / "checkpoints",
    "random_ref":     PROJECT_ROOT / "phase3_outputs" / "random_ref"     / "checkpoints",
    "random_ref_fair": PROJECT_ROOT / "phase3_outputs" / "random_ref_fair" / "checkpoints",
}

EPOCHS = [0, 1, 2]


def build_model(config_path, qwen_path):
    """Build model from config (without loading weights)."""
    from hyperpyyaml import load_hyperpyyaml
    override_dict = {k: None for k in ['flow', 'hift', 'hifigan']}
    override_dict['qwen_pretrain_path'] = str(qwen_path)
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    model = configs['llm']
    return model, configs


def load_checkpoint(model, ckpt_path):
    """Load checkpoint weights into model."""
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model


def build_cv_dataloader(configs, cv_data_path):
    """Build CV dataloader using CosyVoice's dataset infrastructure."""
    from cosyvoice.dataset.dataset import Dataset

    cv_dataset = Dataset(
        str(cv_data_path),
        data_pipeline=configs['data_pipeline'],
        mode='inference',
        shuffle=False,
        partition=False,
    )
    cv_loader = torch.utils.data.DataLoader(
        cv_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )
    return cv_loader


@torch.no_grad()
def evaluate_full_token(model, cv_loader, device):
    """Evaluate model on CV set using ALL speech tokens (no masking).

    Returns:
        dict with total_loss, total_tokens, avg_loss, per_sample results
    """
    from cosyvoice.utils.common import IGNORE_ID as COS_IGNORE_ID, th_accuracy

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_count = 0
    per_utt_results = []

    for batch_idx, batch in enumerate(cv_loader):
        # Move data to device
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        text_token_emb = model.llm.model.model.embed_tokens(text_token)

        # Extract speech tokens
        if 'speech_token' not in batch:
            speech_token, speech_token_len = model.speech_token_extractor.inference(
                batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            speech_token = batch['speech_token'].to(device)
            speech_token_len = batch['speech_token_len'].to(device)
        speech_token_emb = model.speech_embedding(speech_token)

        # Special tokens
        sos_emb = model.llm_embedding.weight[model.sos].reshape(1, 1, -1)
        task_id_emb = model.llm_embedding.weight[model.task_id].reshape(1, 1, -1)

        # Build unistream sequences
        text_token_emb_list = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_list = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        speech_token_emb_list = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)

        lm_targets = []
        lm_inputs = []
        B = text_token.size(0)
        for i in range(B):
            t = torch.tensor(
                [COS_IGNORE_ID] * (1 + int(text_token_len[i].item()))
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
        lm_input = pad_sequence(lm_inputs, batch_first=True, padding_value=COS_IGNORE_ID)
        lm_target = pad_sequence(lm_targets, batch_first=True, padding_value=COS_IGNORE_ID).to(device)

        # LLM forward
        lm_output, _ = model.llm(lm_input, lm_input_len.to(device))
        logits = model.llm_decoder(lm_output)

        # Compute per-token CE loss on ALL speech tokens
        V = logits.size(-1)
        T = logits.size(1)
        flat_logits = logits.view(-1, V)
        flat_target = lm_target.view(-1)
        valid = (flat_target != COS_IGNORE_ID)
        safe_target = flat_target.clone()
        safe_target[~valid] = 0
        flat_loss = F.cross_entropy(flat_logits, safe_target, reduction='none')

        # Per-sample stats
        utts = batch.get('utts', [])
        for i in range(B):
            row_valid = (lm_target[i] != COS_IGNORE_ID)
            n_tokens = row_valid.sum().item()
            if n_tokens > 0:
                row_loss = flat_loss.view(B, T)[i][row_valid].sum().item()
                row_avg = row_loss / n_tokens

                # accuracy
                row_logits = logits[i][row_valid]
                row_targets = lm_target[i][row_valid]
                preds = row_logits.argmax(dim=-1)
                n_correct = (preds == row_targets).sum().item()

                total_loss += row_loss
                total_tokens += n_tokens
                total_correct += n_correct
                total_count += n_tokens

                utt_id = utts[i] if i < len(utts) else f"batch{batch_idx}_s{i}"
                per_utt_results.append({
                    'utt': utt_id,
                    'n_tokens': n_tokens,
                    'loss_sum': row_loss,
                    'avg_loss': row_avg,
                    'accuracy': n_correct / n_tokens,
                })

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    avg_acc = total_correct / total_count if total_count > 0 else float('nan')

    return {
        'total_loss': total_loss,
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'avg_accuracy': avg_acc,
        'n_utterances': len(per_utt_results),
        'per_utt': per_utt_results,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    # Build model structure & dataloader once
    logger.info("Building model from config: %s", CONFIG_PATH)
    model, configs = build_model(CONFIG_PATH, QWEN_PATH)
    model = model.to(device)

    logger.info("Building CV dataloader...")
    cv_loader = build_cv_dataloader(configs, CV_DATA)

    # Evaluate all models × all epochs
    all_results = {}

    for run_name, ckpt_dir in MODELS.items():
        all_results[run_name] = {}
        for epoch in EPOCHS:
            ckpt_path = ckpt_dir / f"epoch_{epoch}_whole.pt"
            if not ckpt_path.exists():
                logger.warning("Checkpoint not found: %s", ckpt_path)
                continue

            logger.info("Evaluating %s epoch %d: %s", run_name, epoch, ckpt_path)
            load_checkpoint(model, ckpt_path)
            result = evaluate_full_token(model, cv_loader, device)

            all_results[run_name][f"epoch_{epoch}"] = {
                'avg_loss': result['avg_loss'],
                'avg_accuracy': result['avg_accuracy'],
                'total_tokens': result['total_tokens'],
                'n_utterances': result['n_utterances'],
            }
            logger.info("  -> avg_loss=%.5f  avg_acc=%.4f  tokens=%d  utts=%d",
                         result['avg_loss'], result['avg_accuracy'],
                         result['total_tokens'], result['n_utterances'])

        # Also save per-utt for final epoch
        if f"epoch_{EPOCHS[-1]}" in all_results[run_name]:
            ckpt_path = ckpt_dir / f"epoch_{EPOCHS[-1]}_whole.pt"
            load_checkpoint(model, ckpt_path)
            result = evaluate_full_token(model, cv_loader, device)
            all_results[run_name]['final_per_utt'] = result['per_utt']

    # Save results
    output_path = OUTPUT_DIR / "unified_eval_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("UNIFIED FULL-TOKEN CV EVALUATION (all speech tokens, no masking)")
    print("=" * 80)
    print(f"\n{'Model':<20} {'Epoch 0':>12} {'Epoch 1':>12} {'Epoch 2':>12} {'Acc (final)':>12}")
    print("-" * 68)
    for run_name in MODELS:
        if run_name not in all_results:
            continue
        vals = []
        for epoch in EPOCHS:
            key = f"epoch_{epoch}"
            if key in all_results[run_name]:
                vals.append(f"{all_results[run_name][key]['avg_loss']:.5f}")
            else:
                vals.append("N/A")
        final_acc = all_results[run_name].get(f"epoch_{EPOCHS[-1]}", {}).get('avg_accuracy', float('nan'))
        print(f"{run_name:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {final_acc:>12.4f}")

    # Rank by final CV loss
    print("\n--- Final CV Loss Ranking (lower is better) ---")
    ranking = []
    for run_name in MODELS:
        final_key = f"epoch_{EPOCHS[-1]}"
        if run_name in all_results and final_key in all_results[run_name]:
            ranking.append((run_name, all_results[run_name][final_key]['avg_loss']))
    ranking.sort(key=lambda x: x[1])
    for i, (name, loss) in enumerate(ranking, 1):
        print(f"  {i}. {name:<20} {loss:.5f}")


if __name__ == '__main__':
    main()
