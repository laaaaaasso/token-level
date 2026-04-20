"""Download full CREMA-D from Hugging Face and build metadata CSV.

Based on examples/prepare_crema_d_hf_sample.py but downloads ALL samples
instead of a random subset.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import audb
import librosa
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download

LOGGER = logging.getLogger("prepare_full_data")

EMO_CODE_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_filename(file_path: str) -> Tuple[int, str, str]:
    """Parse speaker id, sentence code and emotion from CREMA-D filename."""
    name = Path(file_path).name
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected CREMA-D filename: {name}")
    speaker_id = int(parts[0])
    sentence_code = parts[1]
    emo_code = parts[2]
    emotion_label = EMO_CODE_MAP.get(emo_code, "unknown")
    return speaker_id, sentence_code, emotion_label


def build_full_metadata(age_threshold: int, include_disgust: bool) -> pd.DataFrame:
    """Create full metadata frame by combining HF paths with speaker demographics."""
    LOGGER.info("Loading CREMA-D from Hugging Face...")
    hf_ds = load_dataset("MahiA/CREMA-D")
    frames = []
    for split_name in hf_ds:
        frames.append(hf_ds[split_name].to_pandas())
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["path"]).copy()
    LOGGER.info("Total unique samples from HF: %d", len(all_df))

    parsed = all_df["path"].apply(parse_filename)
    all_df["speaker_id"] = parsed.map(lambda x: x[0])
    all_df["text"] = parsed.map(lambda x: x[1])
    all_df["emotion_label"] = parsed.map(lambda x: x[2])

    LOGGER.info("Loading speaker demographics from audb...")
    db = audb.load("crema-d", version="1.3.0", only_metadata=True, full_path=False, verbose=False)
    speaker_meta: Dict[int, Dict[str, object]] = db.schemes["speaker"].labels

    all_df["gender_label"] = all_df["speaker_id"].map(
        lambda sid: str(speaker_meta.get(sid, {}).get("sex", "")).lower()
    )
    all_df["age_num"] = all_df["speaker_id"].map(
        lambda sid: int(speaker_meta.get(sid, {}).get("age"))
        if speaker_meta.get(sid, {}).get("age") is not None
        else None
    )
    all_df["age_label"] = all_df["age_num"].map(
        lambda age: "young" if (pd.notna(age) and int(age) < age_threshold) else "old"
    )

    # Filter emotions
    valid_emos = {"neutral", "happy", "sad", "angry", "fear"}
    if include_disgust:
        valid_emos.add("disgust")
    before = len(all_df)
    all_df = all_df[all_df["emotion_label"].isin(valid_emos)].copy()
    LOGGER.info("Emotion filter: %d -> %d (removed %d 'unknown'/'disgust')", before, len(all_df), before - len(all_df))

    all_df = all_df.rename(columns={"path": "audio_path"})
    cols = ["audio_path", "text", "gender_label", "age_label", "emotion_label", "speaker_id", "age_num"]
    return all_df[cols].reset_index(drop=True)


def validate_audios(outdir: Path, metadata_df: pd.DataFrame, min_dur: float = 0.5, max_dur: float = 15.0) -> pd.DataFrame:
    """Validate audio files exist and have reasonable duration. Return cleaned df."""
    valid_rows = []
    bad_count = 0
    missing_count = 0
    for idx, row in metadata_df.iterrows():
        audio_path = outdir / row["audio_path"]
        if not audio_path.exists():
            missing_count += 1
            continue
        try:
            dur = librosa.get_duration(path=str(audio_path))
            if dur < min_dur or dur > max_dur:
                LOGGER.debug("Duration out of range (%.2fs): %s", dur, audio_path.name)
                bad_count += 1
                continue
            valid_rows.append(idx)
        except Exception as e:
            LOGGER.warning("Cannot read audio %s: %s", audio_path.name, e)
            bad_count += 1

    LOGGER.info("Validation: %d valid, %d missing, %d bad/out-of-range (total %d)",
                len(valid_rows), missing_count, bad_count, len(metadata_df))
    return metadata_df.loc[valid_rows].reset_index(drop=True)


def compute_stats(metadata_df: pd.DataFrame) -> dict:
    """Compute data distribution statistics."""
    stats = {
        "total_samples": len(metadata_df),
        "num_speakers": int(metadata_df["speaker_id"].nunique()),
        "emotion_distribution": metadata_df["emotion_label"].value_counts().to_dict(),
        "gender_distribution": metadata_df["gender_label"].value_counts().to_dict(),
        "age_distribution": metadata_df["age_label"].value_counts().to_dict(),
        "text_distribution": metadata_df["text"].value_counts().to_dict(),
        "samples_per_speaker": {
            "mean": float(metadata_df.groupby("speaker_id").size().mean()),
            "min": int(metadata_df.groupby("speaker_id").size().min()),
            "max": int(metadata_df.groupby("speaker_id").size().max()),
        },
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download full CREMA-D and prepare metadata.")
    parser.add_argument("--outdir", default="crema_data_full", help="Output directory.")
    parser.add_argument("--age-threshold", type=int, default=40, help="Age < threshold => young.")
    parser.add_argument("--include-disgust", action="store_true", help="Keep disgust emotion samples.")
    parser.add_argument("--max-workers", type=int, default=8, help="Download workers.")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if audios already exist.")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Min audio duration (sec).")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Max audio duration (sec).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build metadata
    LOGGER.info("Building full CREMA-D metadata...")
    full_df = build_full_metadata(
        age_threshold=args.age_threshold,
        include_disgust=args.include_disgust,
    )
    LOGGER.info("Full metadata: %d samples", len(full_df))

    # Step 2: Download all audios
    if args.skip_download:
        LOGGER.info("Skipping download (--skip-download).")
    else:
        allow_patterns = full_df["audio_path"].tolist()
        LOGGER.info("Downloading %d audio files from HF...", len(allow_patterns))
        snapshot_download(
            repo_id="MahiA/CREMA-D",
            repo_type="dataset",
            local_dir=str(outdir),
            allow_patterns=allow_patterns,
            max_workers=args.max_workers,
        )
        LOGGER.info("Download complete.")

    # Step 3: Validate audios
    LOGGER.info("Validating audio files...")
    clean_df = validate_audios(outdir, full_df, min_dur=args.min_duration, max_dur=args.max_duration)

    # Step 4: Save metadata and stats
    metadata_path = outdir / "metadata.csv"
    clean_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved metadata: %s (%d samples)", metadata_path, len(clean_df))

    stats = compute_stats(clean_df)
    stats_path = outdir / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved stats: %s", stats_path)

    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("SUMMARY")
    LOGGER.info("  Total samples: %d", stats["total_samples"])
    LOGGER.info("  Speakers: %d", stats["num_speakers"])
    LOGGER.info("  Emotions: %s", stats["emotion_distribution"])
    LOGGER.info("  Gender: %s", stats["gender_distribution"])
    LOGGER.info("  Age: %s", stats["age_distribution"])
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
