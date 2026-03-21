"""Download CREMA-D from Hugging Face and build a sampled metadata CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import audb
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download


LOGGER = logging.getLogger("prepare_crema_d_hf_sample")


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


def build_metadata_frame(age_threshold: int, include_disgust: bool) -> pd.DataFrame:
    """Create full metadata frame by combining HF paths with speaker demographics."""
    hf_ds = load_dataset("MahiA/CREMA-D")
    all_df = pd.concat([hf_ds["train"].to_pandas(), hf_ds["test"].to_pandas()], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["path"]).copy()

    parsed = all_df["path"].apply(parse_filename)
    all_df["speaker_id"] = parsed.map(lambda x: x[0])
    all_df["text"] = parsed.map(lambda x: x[1])
    all_df["emotion_label"] = parsed.map(lambda x: x[2])

    db = audb.load("crema-d", version="1.3.0", only_metadata=True, full_path=False, verbose=False)
    speaker_meta: Dict[int, Dict[str, object]] = db.schemes["speaker"].labels

    all_df["gender_label"] = all_df["speaker_id"].map(
        lambda sid: str(speaker_meta.get(sid, {}).get("sex", "")).lower()
    )
    all_df["age_num"] = all_df["speaker_id"].map(
        lambda sid: int(speaker_meta.get(sid, {}).get("age")) if speaker_meta.get(sid, {}).get("age") is not None else None
    )
    all_df["age_label"] = all_df["age_num"].map(
        lambda age: "young" if (pd.notna(age) and int(age) < age_threshold) else "old"
    )

    valid_emos = {"neutral", "happy", "sad", "angry", "fear"}
    if include_disgust:
        valid_emos.add("disgust")
    all_df = all_df[all_df["emotion_label"].isin(valid_emos)].copy()

    all_df = all_df.rename(columns={"path": "audio_path"})
    cols = ["audio_path", "text", "gender_label", "age_label", "emotion_label", "speaker_id", "age_num"]
    return all_df[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="crema_data_1000", help="Output directory.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--age-threshold", type=int, default=40, help="Age < threshold => young else old.")
    parser.add_argument("--include-disgust", action="store_true", help="Keep disgust emotion samples.")
    parser.add_argument("--max-workers", type=int, default=8, help="Download workers for HF snapshot.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building CREMA-D metadata index from Hugging Face and audb demographics...")
    full_df = build_metadata_frame(age_threshold=args.age_threshold, include_disgust=args.include_disgust)
    total = len(full_df)
    if total < args.n_samples:
        raise ValueError(f"Requested {args.n_samples} samples, but only {total} available after filtering.")

    sampled = full_df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)
    allow_patterns = sampled["audio_path"].tolist()

    LOGGER.info("Downloading %s selected audios from HF dataset repository...", len(allow_patterns))
    snapshot_download(
        repo_id="MahiA/CREMA-D",
        repo_type="dataset",
        local_dir=str(outdir),
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
    )

    metadata_path = outdir / "metadata.csv"
    sampled.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved sampled metadata: %s", metadata_path)
    LOGGER.info("Emotion distribution:\n%s", sampled["emotion_label"].value_counts().to_string())
    LOGGER.info("Gender distribution:\n%s", sampled["gender_label"].value_counts().to_string())
    LOGGER.info("Age distribution:\n%s", sampled["age_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
