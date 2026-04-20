#!/usr/bin/env python3
"""Stage D2: Compute objective generation metrics.

Metrics:
  1. ASR-WER: Use Whisper to transcribe generated audio, compute WER against ground truth text
  2. Speaker similarity: Use CosyVoice2's campplus speaker encoder to compute cosine similarity
     between reference audio and generated audio speaker embeddings
  3. Duration analysis: Compare generated vs reference audio durations

Usage:
    conda activate cosyvoice2
    python stage_d_analysis/compute_metrics.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_ROOT = Path("/data/zhenghao/repos/CosyVoice")
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

OUTPUT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = OUTPUT_DIR / "generated_audio"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B"

MODELS = ['pretrained', 'selective', 'full_baseline', 'random_mask', 'random_ref_fair']

# Ground truth texts for each sentence code
GT_TEXTS = {
    'IEO': "it's eleven o'clock",
    'TIE': "that is exactly what happened",
    'IOM': "i'm on my way to the meeting",
    'IWW': "i wonder what this is about",
    'DFA': "don't forget a jacket",
    'ITH': "i think i have a doctor's appointment",
}


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()

    # Dynamic programming for edit distance
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[n][m] / max(n, 1)


def extract_speaker_embedding(wav_path, campplus_session, target_sr=16000):
    """Extract speaker embedding using campplus ONNX model."""
    import onnxruntime
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Compute fbank features (80-dim, matching campplus input)
    import torchaudio.compliance.kaldi as kaldi
    feat = kaldi.fbank(waveform, num_mel_bins=80, sample_frequency=target_sr,
                       frame_length=25, frame_shift=10)  # [T, 80]
    feat = feat - feat.mean(dim=0, keepdim=True)  # CMN
    feat = feat.unsqueeze(0).numpy()  # [1, T, 80]

    # Run campplus
    outputs = campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat})
    embedding = outputs[0]  # [1, D]
    return embedding[0]


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    import whisper
    import onnxruntime

    # Load Whisper model
    logger.info("Loading Whisper model (base.en)...")
    whisper_model = whisper.load_model("base.en")

    # Load campplus for speaker similarity
    campplus_path = str(PRETRAINED_DIR / "campplus.onnx")
    logger.info("Loading campplus from %s", campplus_path)
    campplus_session = onnxruntime.InferenceSession(campplus_path,
                                                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Load test configs
    with open(OUTPUT_DIR / "test_configs.json") as f:
        test_configs = json.load(f)

    all_results = {}

    for model_name in MODELS:
        logger.info("=" * 50)
        logger.info("Evaluating model: %s", model_name)
        model_dir = AUDIO_DIR / model_name
        model_results = []

        for cfg in test_configs:
            label = cfg['label']
            sentence_code = cfg['sentence_code']
            gt_text = GT_TEXTS.get(sentence_code, "")
            gen_path = model_dir / f"{label}.wav"
            ref_path = cfg['prompt_wav']

            if not gen_path.exists():
                logger.warning("  Missing: %s", gen_path)
                continue

            result = {'label': label, 'emotion': cfg['emotion'],
                      'sentence_code': sentence_code}

            # 1. ASR-WER
            try:
                asr_result = whisper_model.transcribe(str(gen_path), language='en')
                hyp_text = asr_result['text'].strip()
                wer = compute_wer(gt_text, hyp_text)
                result['asr_text'] = hyp_text
                result['gt_text'] = gt_text
                result['wer'] = wer
                logger.info("  [%s] WER=%.3f  hyp='%s'", label, wer, hyp_text)
            except Exception as e:
                result['wer'] = None
                result['asr_error'] = str(e)
                logger.error("  [%s] ASR failed: %s", label, e)

            # 2. Speaker similarity
            try:
                gen_emb = extract_speaker_embedding(str(gen_path), campplus_session)
                ref_emb = extract_speaker_embedding(ref_path, campplus_session)
                spk_sim = cosine_similarity(gen_emb, ref_emb)
                result['speaker_similarity'] = spk_sim
                logger.info("  [%s] SpkSim=%.4f", label, spk_sim)
            except Exception as e:
                result['speaker_similarity'] = None
                result['spk_error'] = str(e)
                logger.error("  [%s] SpkSim failed: %s", label, e)

            # 3. Duration
            try:
                gen_wav, gen_sr = torchaudio.load(str(gen_path))
                ref_wav, ref_sr = torchaudio.load(ref_path)
                gen_dur = gen_wav.shape[1] / gen_sr
                ref_dur = ref_wav.shape[1] / ref_sr
                result['gen_duration'] = float(gen_dur)
                result['ref_duration'] = float(ref_dur)
                result['duration_ratio'] = float(gen_dur / ref_dur) if ref_dur > 0 else None
            except Exception as e:
                logger.error("  [%s] Duration failed: %s", label, e)

            model_results.append(result)

        all_results[model_name] = model_results

    # Save detailed results
    results_path = OUTPUT_DIR / "objective_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info("Detailed results saved to %s", results_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("STAGE D2: OBJECTIVE METRICS SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Avg WER':>10} {'Avg SpkSim':>12} {'Avg DurRatio':>12} {'Samples':>8}")
    print("-" * 62)
    for model_name in MODELS:
        results = all_results.get(model_name, [])
        wers = [r['wer'] for r in results if r.get('wer') is not None]
        sims = [r['speaker_similarity'] for r in results if r.get('speaker_similarity') is not None]
        durs = [r['duration_ratio'] for r in results if r.get('duration_ratio') is not None]
        avg_wer = np.mean(wers) if wers else float('nan')
        avg_sim = np.mean(sims) if sims else float('nan')
        avg_dur = np.mean(durs) if durs else float('nan')
        print(f"{model_name:<20} {avg_wer:>10.3f} {avg_sim:>12.4f} {avg_dur:>12.3f} {len(results):>8}")

    # Per-emotion breakdown
    print(f"\n--- Per-Emotion WER ---")
    emotions = sorted(set(r['emotion'] for results in all_results.values() for r in results))
    print(f"{'Model':<20}", end="")
    for e in emotions:
        print(f" {e:>10}", end="")
    print()
    print("-" * (20 + 11 * len(emotions)))
    for model_name in MODELS:
        results = all_results.get(model_name, [])
        print(f"{model_name:<20}", end="")
        for e in emotions:
            wers = [r['wer'] for r in results if r.get('wer') is not None and r['emotion'] == e]
            avg = np.mean(wers) if wers else float('nan')
            print(f" {avg:>10.3f}", end="")
        print()

    print(f"\n--- Per-Emotion Speaker Similarity ---")
    print(f"{'Model':<20}", end="")
    for e in emotions:
        print(f" {e:>10}", end="")
    print()
    print("-" * (20 + 11 * len(emotions)))
    for model_name in MODELS:
        results = all_results.get(model_name, [])
        print(f"{model_name:<20}", end="")
        for e in emotions:
            sims = [r['speaker_similarity'] for r in results if r.get('speaker_similarity') is not None and r['emotion'] == e]
            avg = np.mean(sims) if sims else float('nan')
            print(f" {avg:>10.4f}", end="")
        print()


if __name__ == '__main__':
    main()
