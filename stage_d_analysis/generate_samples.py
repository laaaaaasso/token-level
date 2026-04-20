#!/usr/bin/env python3
"""Stage D1: Generate speech samples from all model checkpoints.

Uses CosyVoice2's standard inference pipeline with cross_lingual mode.
Only the LLM weights are swapped; flow + hift remain pretrained.

Generates from a fixed set of test texts × reference audios,
so all models produce samples for the same conditions.

Usage:
    conda activate cosyvoice2
    python stage_d_analysis/generate_samples.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_ROOT = Path("/data/zhenghao/repos/CosyVoice")
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CREMA-D sentence map
SENTENCE_MAP = {
    'IEO': "It's eleven o'clock.",
    'TIE': "That is exactly what happened.",
    'IOM': "I'm on my way to the meeting.",
    'IWW': "I wonder what this is about.",
    'DFA': "Don't forget a jacket.",
    'ITH': "I think I have a doctor's appointment.",
}

# Test configurations: pick diverse emotions × sentences × speakers
# We'll pick reference audios that cover different emotions
TEST_CONFIGS = [
    # (test_text, prompt_wav_path, label)
    # Use 4 diverse sentences × 3 emotions = 12 samples per model
]

# Models to evaluate (final epoch checkpoint)
MODELS = {
    'pretrained':     str(PRETRAINED_DIR / "llm.pt"),
    'selective':      str(PROJECT_ROOT / "phase3_outputs" / "selective"      / "checkpoints" / "epoch_2_whole.pt"),
    'full_baseline':  str(PROJECT_ROOT / "phase3_outputs" / "full_baseline"  / "checkpoints" / "epoch_2_whole.pt"),
    'random_mask':    str(PROJECT_ROOT / "phase3_outputs" / "random_mask"    / "checkpoints" / "epoch_2_whole.pt"),
    'random_ref_fair': str(PROJECT_ROOT / "phase3_outputs" / "random_ref_fair" / "checkpoints" / "epoch_2_whole.pt"),
}


def find_reference_audios():
    """Find diverse reference audios from CREMA-D."""
    import pandas as pd
    meta = pd.read_csv(PROJECT_ROOT / "crema_data_1000" / "metadata.csv")
    audio_dir = PROJECT_ROOT / "crema_data_1000" / "audios"

    configs = []
    # Pick samples covering: angry, happy, sad, neutral × different sentences
    targets = [
        ('IEO', 'angry',  "It's eleven o'clock."),
        ('IEO', 'happy',  "It's eleven o'clock."),
        ('IEO', 'sad',    "It's eleven o'clock."),
        ('IOM', 'angry',  "I'm on my way to the meeting."),
        ('IOM', 'happy',  "I'm on my way to the meeting."),
        ('IOM', 'neutral',"I'm on my way to the meeting."),
        ('DFA', 'angry',  "Don't forget a jacket."),
        ('DFA', 'sad',    "Don't forget a jacket."),
        ('ITH', 'happy',  "I think I have a doctor's appointment."),
        ('ITH', 'neutral',"I think I have a doctor's appointment."),
        ('TIE', 'angry',  "That is exactly what happened."),
        ('TIE', 'sad',    "That is exactly what happened."),
    ]

    for sent_code, emotion, full_text in targets:
        # Find a matching audio in metadata
        matches = meta[(meta['text'] == sent_code) & (meta['emotion_label'] == emotion)]
        if len(matches) == 0:
            logger.warning("No match for %s/%s, skipping", sent_code, emotion)
            continue
        row = matches.iloc[0]
        wav_path = str(audio_dir / Path(row['audio_path']).name)
        if not os.path.exists(wav_path):
            logger.warning("Audio not found: %s", wav_path)
            continue
        label = f"{sent_code}_{emotion}_{row['speaker_id']}"
        configs.append({
            'text': full_text,
            'prompt_wav': wav_path,
            'label': label,
            'sentence_code': sent_code,
            'emotion': emotion,
            'speaker_id': int(row['speaker_id']),
        })

    logger.info("Found %d test configurations", len(configs))
    return configs


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice2

    logger.info("Loading CosyVoice2 from %s", PRETRAINED_DIR)
    cosyvoice = CosyVoice2(model_dir=str(PRETRAINED_DIR))
    sample_rate = cosyvoice.sample_rate

    # Get test configs
    test_configs = find_reference_audios()
    if not test_configs:
        logger.error("No test configs found!")
        return

    # Save test config for reference
    config_path = OUTPUT_DIR.parent / "test_configs.json"
    with open(config_path, 'w') as f:
        json.dump(test_configs, f, indent=2)
    logger.info("Saved %d test configs to %s", len(test_configs), config_path)

    # Copy reference audios
    ref_dir = OUTPUT_DIR / "reference"
    ref_dir.mkdir(exist_ok=True)
    for cfg in test_configs:
        src = cfg['prompt_wav']
        dst = ref_dir / f"{cfg['label']}_ref.wav"
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    results = {}

    for model_name, ckpt_path in MODELS.items():
        logger.info("=" * 60)
        logger.info("Generating with model: %s", model_name)
        logger.info("  checkpoint: %s", ckpt_path)

        # Load LLM weights (filter out non-model keys like 'epoch', 'step')
        state_dict = torch.load(ckpt_path, map_location=cosyvoice.model.device)
        state_dict = {k: v for k, v in state_dict.items()
                      if k not in ('epoch', 'step', 'optimizer', 'scheduler')}
        cosyvoice.model.llm.load_state_dict(state_dict, strict=True)
        cosyvoice.model.llm.eval()
        logger.info("  LLM weights loaded")

        model_dir = OUTPUT_DIR / model_name
        model_dir.mkdir(exist_ok=True)
        model_results = []

        for cfg in test_configs:
            label = cfg['label']
            text = cfg['text']
            prompt_wav = cfg['prompt_wav']

            out_path = model_dir / f"{label}.wav"
            logger.info("  Generating: %s -> %s", label, out_path.name)

            start_time = time.time()
            try:
                for i, output in enumerate(cosyvoice.inference_cross_lingual(
                        text, prompt_wav, stream=False, text_frontend=False)):
                    tts_speech = output['tts_speech']
                    torchaudio.save(str(out_path), tts_speech, sample_rate)
                    break  # Take first output only

                gen_time = time.time() - start_time
                duration = tts_speech.shape[1] / sample_rate
                rtf = gen_time / duration if duration > 0 else 0

                model_results.append({
                    'label': label,
                    'text': text,
                    'emotion': cfg['emotion'],
                    'duration': float(duration),
                    'gen_time': float(gen_time),
                    'rtf': float(rtf),
                    'status': 'ok',
                })
                logger.info("    -> %.2fs, duration=%.2fs, RTF=%.3f", gen_time, duration, rtf)
            except Exception as e:
                logger.error("    -> FAILED: %s", e)
                model_results.append({
                    'label': label,
                    'text': text,
                    'emotion': cfg['emotion'],
                    'status': 'error',
                    'error': str(e),
                })

        results[model_name] = model_results

    # Save generation results
    results_path = OUTPUT_DIR.parent / "generation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Print summary
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    for model_name in MODELS:
        if model_name not in results:
            continue
        ok = sum(1 for r in results[model_name] if r['status'] == 'ok')
        total = len(results[model_name])
        avg_rtf = sum(r.get('rtf', 0) for r in results[model_name] if r['status'] == 'ok') / max(ok, 1)
        avg_dur = sum(r.get('duration', 0) for r in results[model_name] if r['status'] == 'ok') / max(ok, 1)
        print(f"  {model_name:<20} {ok}/{total} ok  avg_dur={avg_dur:.2f}s  avg_rtf={avg_rtf:.3f}")
    print(f"\nAudio saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
