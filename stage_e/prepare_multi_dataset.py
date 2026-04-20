"""
Step 1: Sample ~30 hours of speech data from HiFi-TTS (clean + other) datasets.

Since Emilia is gated (requires auth) and the environment has no HF token,
we use HiFi-TTS clean (~20h) + HiFi-TTS other (~10h) as two distinct sources:
  - clean: high-quality studio recordings
  - other: more varied recording conditions

All audio is resampled to 16000 Hz (matching CosyVoice2 pipeline).

Output:
  multi_dataset_30h/audios/       -- wav files
  multi_dataset_30h/metadata.csv  -- unified manifest
  multi_dataset_30h/data_stats.json -- statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

LOGGER = logging.getLogger("prepare_multi_dataset")

HF_MIRROR = "https://hf-mirror.com"
TARGET_SR = 16000
MIN_DUR = 0.5
MAX_DUR = 15.0


def setup_hf_mirror():
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    LOGGER.info("Using HF mirror: %s", HF_MIRROR)


def stream_hifitts(config, split, max_hours, max_samples_per_speaker=0, seed=42):
    from datasets import load_dataset

    LOGGER.info("Streaming HiFi-TTS config=%s split=%s (target %.1f hours)...",
                config, split, max_hours)

    ds = load_dataset("MikhailT/hifi-tts", config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    samples = []
    total_dur = 0.0
    max_dur_sec = max_hours * 3600
    speaker_counts = defaultdict(int)
    skipped_dur = 0
    skipped_speaker = 0

    for i, example in enumerate(ds):
        if total_dur >= max_dur_sec:
            break

        audio = example.get("audio", {})
        array = audio.get("array")
        sr = audio.get("sampling_rate", 44100)
        duration = example.get("duration", len(array) / sr if array is not None else 0)
        speaker = str(example.get("speaker", "unknown"))
        text = example.get("text", "")

        if duration < MIN_DUR or duration > MAX_DUR:
            skipped_dur += 1
            continue

        if max_samples_per_speaker > 0 and speaker_counts[speaker] >= max_samples_per_speaker:
            skipped_speaker += 1
            continue

        samples.append({
            "audio_array": np.array(array, dtype=np.float32),
            "original_sr": sr,
            "text": text,
            "speaker_id": speaker,
            "duration": duration,
            "file": example.get("file", "sample_%d" % i),
        })
        total_dur += duration
        speaker_counts[speaker] += 1

        if (i + 1) % 500 == 0:
            LOGGER.info("  [%s-%s] Processed %d, collected %d, %.1f hours",
                        config, split, i + 1, len(samples), total_dur / 3600)

    LOGGER.info("  [%s-%s] Done: %d samples, %.2f hours. Skipped: %d dur, %d speaker",
                config, split, len(samples), total_dur / 3600, skipped_dur, skipped_speaker)
    LOGGER.info("  Speakers: %d unique", len(speaker_counts))
    return samples


def resample_and_save(samples, audio_dir, prefix, target_sr=TARGET_SR):
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for i, s in enumerate(samples):
        array = s["audio_array"]
        orig_sr = s["original_sr"]

        if orig_sr != target_sr:
            array = librosa.resample(array, orig_sr=orig_sr, target_sr=target_sr)

        duration = len(array) / target_sr
        speaker = s["speaker_id"]
        fname = "%s_%s_%06d.wav" % (prefix, speaker, i)
        fpath = audio_dir / fname

        sf.write(str(fpath), array, target_sr)

        metadata.append({
            "audio_path": "audios/%s" % fname,
            "text": s["text"],
            "speaker_id": "%s_%s" % (prefix, speaker),
            "dataset_source": prefix,
            "duration_sec": round(duration, 3),
        })

        if (i + 1) % 1000 == 0:
            LOGGER.info("  Saved %d / %d files", i + 1, len(samples))

    LOGGER.info("  Saved %d wav files to %s", len(metadata), audio_dir)
    return metadata


def validate_audios(base_dir, metadata):
    valid = []
    bad = 0
    for m in metadata:
        fpath = base_dir / m["audio_path"]
        if not fpath.exists():
            bad += 1
            continue
        try:
            dur = librosa.get_duration(path=str(fpath))
            if dur < MIN_DUR or dur > MAX_DUR:
                bad += 1
                continue
            m["duration_sec"] = round(dur, 3)
            valid.append(m)
        except Exception as e:
            LOGGER.warning("Cannot read %s: %s", fpath, e)
            bad += 1

    LOGGER.info("Validation: %d valid, %d bad (total %d)", len(valid), bad, len(metadata))
    return valid


def compute_stats(metadata_df):
    durations = metadata_df["duration_sec"]
    stats = {
        "total_samples": len(metadata_df),
        "total_hours": round(durations.sum() / 3600, 2),
        "num_speakers": int(metadata_df["speaker_id"].nunique()),
        "num_datasets": int(metadata_df["dataset_source"].nunique()),
        "duration_stats": {
            "min": round(float(durations.min()), 3),
            "max": round(float(durations.max()), 3),
            "mean": round(float(durations.mean()), 3),
            "median": round(float(durations.median()), 3),
            "stdev": round(float(durations.std()), 3),
            "p5": round(float(np.percentile(durations, 5)), 3),
            "p95": round(float(np.percentile(durations, 95)), 3),
        },
        "per_dataset": {},
        "samples_per_speaker_stats": {
            "min": int(metadata_df.groupby("speaker_id").size().min()),
            "max": int(metadata_df.groupby("speaker_id").size().max()),
            "mean": round(float(metadata_df.groupby("speaker_id").size().mean()), 1),
        },
    }

    for src, grp in metadata_df.groupby("dataset_source"):
        d = grp["duration_sec"]
        stats["per_dataset"][src] = {
            "samples": len(grp),
            "hours": round(d.sum() / 3600, 2),
            "speakers": int(grp["speaker_id"].nunique()),
            "duration_mean": round(float(d.mean()), 3),
            "duration_stdev": round(float(d.std()), 3),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-dataset ~30h speech data")
    parser.add_argument("--outdir", default="multi_dataset_30h", help="Output directory")
    parser.add_argument("--hifitts-clean-hours", type=float, default=20.0)
    parser.add_argument("--hifitts-other-hours", type=float, default=10.0)
    parser.add_argument("--max-samples-per-speaker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audio_dir = outdir / "audios"

    if args.skip_download:
        LOGGER.info("Skip download mode")
        meta_path = outdir / "metadata.csv"
        if not meta_path.exists():
            LOGGER.error("No metadata.csv found at %s", meta_path)
            return
        df = pd.read_csv(meta_path)
    else:
        setup_hf_mirror()
        all_metadata = []

        # 1) HiFi-TTS clean
        LOGGER.info("=" * 60)
        LOGGER.info("Streaming HiFi-TTS clean (target %.1fh)", args.hifitts_clean_hours)
        LOGGER.info("=" * 60)
        clean_samples = stream_hifitts(
            config="clean", split="train",
            max_hours=args.hifitts_clean_hours,
            max_samples_per_speaker=args.max_samples_per_speaker,
            seed=args.seed,
        )

        LOGGER.info("Saving HiFi-TTS clean audio (resample to %dHz)", args.target_sr)
        clean_meta = resample_and_save(clean_samples, audio_dir, "hifitts_clean", args.target_sr)
        all_metadata.extend(clean_meta)
        del clean_samples

        # 2) HiFi-TTS other
        LOGGER.info("=" * 60)
        LOGGER.info("Streaming HiFi-TTS other (target %.1fh)", args.hifitts_other_hours)
        LOGGER.info("=" * 60)
        other_samples = stream_hifitts(
            config="other", split="train",
            max_hours=args.hifitts_other_hours,
            max_samples_per_speaker=args.max_samples_per_speaker,
            seed=args.seed,
        )

        LOGGER.info("Saving HiFi-TTS other audio (resample to %dHz)", args.target_sr)
        other_meta = resample_and_save(other_samples, audio_dir, "hifitts_other", args.target_sr)
        all_metadata.extend(other_meta)
        del other_samples

        # 3) Save metadata
        df = pd.DataFrame(all_metadata)
        meta_path = outdir / "metadata.csv"
        df.to_csv(meta_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved metadata: %s (%d samples)", meta_path, len(df))

    # 4) Validate
    LOGGER.info("=" * 60)
    LOGGER.info("Validating audio files...")
    LOGGER.info("=" * 60)
    valid_meta = validate_audios(outdir, df.to_dict("records"))
    df = pd.DataFrame(valid_meta)

    meta_path = outdir / "metadata.csv"
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved cleaned metadata: %s (%d samples)", meta_path, len(df))

    # 5) Stats
    stats = compute_stats(df)
    stats_path = outdir / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved stats: %s", stats_path)

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("SUMMARY")
    LOGGER.info("  Total samples: %d", stats["total_samples"])
    LOGGER.info("  Total hours: %.2f", stats["total_hours"])
    LOGGER.info("  Speakers: %d", stats["num_speakers"])
    LOGGER.info("  Duration: mean=%.2fs, median=%.2fs, p5=%.2fs, p95=%.2fs",
                stats["duration_stats"]["mean"],
                stats["duration_stats"]["median"],
                stats["duration_stats"]["p5"],
                stats["duration_stats"]["p95"])
    for src, info in stats["per_dataset"].items():
        LOGGER.info("  [%s] %d samples, %.2fh, %d speakers",
                    src, info["samples"], info["hours"], info["speakers"])
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
