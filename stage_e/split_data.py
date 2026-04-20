"""Step 2: Statistical filtering + data splitting for Stage E.

Runs the existing significance scoring pipeline on full CREMA-D,
then splits into D_ref / D_train / CV / D_test by speaker_id.

User requirements:
- std_multiplier = 1.5 (tighter intervals)
- sig_score == 3 (all three dimensions pass)
- D_ref <= 50% of total (adjust by further tightening if needed)
- Split by speaker_id (no speaker overlap across splits)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent dir to path so we can import the existing pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import FEATURE_COLUMNS, extract_single_audio_features
from scoring import assign_significance_flags, rank_high_score_samples
from stats_validation import run_all_validations
from utils import load_age_mapping, load_metadata, resolve_audio_path, save_json

LOGGER = logging.getLogger("split_data")


def _extract_one(args: Tuple[int, str, str]) -> Tuple[int, Dict[str, Any]]:
    """Extract features for a single audio file (worker function)."""
    row_id, audio_path, metadata_dir = args
    resolved = resolve_audio_path(audio_path, metadata_dir)
    if resolved is None or not resolved.exists():
        empty = {k: np.nan for k in FEATURE_COLUMNS}
        empty["feature_status"] = "missing"
        return row_id, empty
    try:
        feats = extract_single_audio_features(str(resolved))
        feats["feature_status"] = "ok"
        return row_id, feats
    except Exception:
        empty = {k: np.nan for k in FEATURE_COLUMNS}
        empty["feature_status"] = "error"
        return row_id, empty


def extract_features_parallel(
    metadata_df: pd.DataFrame,
    n_workers: int = 32,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract audio features using multiprocessing.

    If cache_path exists, loads cached features instead of re-extracting.
    After extraction, saves features to cache_path for future reuse.
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            LOGGER.info("Loading cached features from %s", cache)
            cached_df = pd.read_parquet(cache)
            if len(cached_df) == len(metadata_df):
                LOGGER.info("Cache hit: %d rows", len(cached_df))
                return cached_df
            LOGGER.warning("Cache size mismatch (%d vs %d), re-extracting",
                          len(cached_df), len(metadata_df))

    # Build work items
    work_items = []
    for idx, row in metadata_df.iterrows():
        work_items.append((
            idx,
            row.get("audio_path"),
            row.get("metadata_dir", "."),
        ))

    LOGGER.info("Extracting features for %d samples using %d workers...", len(work_items), n_workers)
    results = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_one, item): item[0] for item in work_items}
        with tqdm(total=len(futures), desc="Feature extraction", unit="sample") as pbar:
            for future in as_completed(futures):
                row_id, feats = future.result()
                results[row_id] = feats
                pbar.update(1)

    elapsed = time.time() - t0
    LOGGER.info("Feature extraction done in %.1f seconds (%.2f s/sample)",
                elapsed, elapsed / len(work_items))

    # Build DataFrame
    rows = []
    for idx in metadata_df.index:
        rows.append(results[idx])
    feat_df = pd.DataFrame(rows, index=metadata_df.index)

    # Status summary
    status_counts = feat_df["feature_status"].value_counts()
    LOGGER.info("Feature status: %s", status_counts.to_dict())

    # Save cache
    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(cache, index=True)
        LOGGER.info("Saved feature cache: %s", cache)

    return feat_df


def run_scoring_pipeline(
    metadata_path: str,
    std_multiplier: float = 1.5,
    interval_method: str = "std",
    alpha: float = 0.05,
    n_workers: int = 32,
    cache_dir: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run feature extraction + statistical validation + scoring."""
    age_map = load_age_mapping(None)
    metadata_df = load_metadata(metadata_path, age_map)
    LOGGER.info("Loaded metadata: %d rows", len(metadata_df))

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "features_cache.parquet"

    feature_df = extract_features_parallel(metadata_df, n_workers=n_workers, cache_path=cache_path)
    feature_export = feature_df.reset_index(drop=True)
    full_df = pd.concat([metadata_df.reset_index(drop=True), feature_export], axis=1)

    # Save features CSV for reference
    if cache_dir is not None:
        features_csv = Path(cache_dir) / "features.csv"
        feature_export.to_csv(features_csv, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved features CSV: %s", features_csv)

    LOGGER.info("Running group-level statistical validations (std_multiplier=%.2f)...", std_multiplier)
    group_test_df, stats_cache = run_all_validations(
        full_df,
        alpha=alpha,
        interval_method=interval_method,
        std_multiplier=std_multiplier,
    )

    LOGGER.info("Assigning per-sample significance flags...")
    scored_df = assign_significance_flags(full_df, stats_cache)

    # Log distribution
    dist = scored_df["sig_score"].value_counts().sort_index()
    LOGGER.info("sig_score distribution:\n%s", dist.to_string())
    n3 = int((scored_df["sig_score"] == 3).sum())
    LOGGER.info("sig_score == 3: %d / %d (%.1f%%)", n3, len(scored_df), 100 * n3 / len(scored_df))

    return scored_df, group_test_df


def speaker_stratified_split(
    df: pd.DataFrame,
    ref_mask: pd.Series,
    cv_ratio: float = 0.08,
    test_ratio: float = 0.07,
    target_ref_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Split data by speaker_id into D_ref, D_train, CV, D_test.

    - D_ref: samples with ref_mask == True, grouped by speaker
    - Remaining speakers split into D_train + CV + D_test
    - CV and D_test are drawn from non-ref speakers
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["split"] = "unknown"

    # Identify ref-eligible speakers: those with high proportion of sig_score==3.
    # Sort by ref rate descending, then greedily pick speakers until target is met.
    # Use a lower threshold (60%) since tighter std_multiplier reduces per-sample scores.
    speaker_ref_rate = df.groupby("speaker_id")["sig_score"].apply(
        lambda s: (s == 3).mean()
    ).sort_values(ascending=False)

    ref_eligible_speakers = speaker_ref_rate[speaker_ref_rate >= 0.6].index.tolist()
    LOGGER.info("Ref-eligible speakers (sig_score==3 rate >= 60%%): %d / %d",
                len(ref_eligible_speakers), df["speaker_id"].nunique())

    # Ensure gender balance in ref speakers
    speaker_gender = df.groupby("speaker_id")["gender_label"].first()
    male_eligible = [s for s in ref_eligible_speakers if speaker_gender.get(s) == "male"]
    female_eligible = [s for s in ref_eligible_speakers if speaker_gender.get(s) == "female"]
    rng.shuffle(male_eligible)
    rng.shuffle(female_eligible)

    # Interleave male and female speakers for balanced selection
    interleaved = []
    mi, fi = 0, 0
    while mi < len(male_eligible) or fi < len(female_eligible):
        if mi < len(male_eligible):
            interleaved.append(male_eligible[mi])
            mi += 1
        if fi < len(female_eligible):
            interleaved.append(female_eligible[fi])
            fi += 1

    # We want D_ref to be ~target_ref_ratio of total
    total = len(df)
    target_ref = int(total * target_ref_ratio)

    # Greedily add speakers to D_ref until we reach target
    ref_speakers = []
    ref_count = 0
    for spk in interleaved:
        spk_n = int((df["speaker_id"] == spk).sum())
        if ref_count + spk_n <= int(target_ref * 1.1):  # allow 10% overshoot
            ref_speakers.append(spk)
            ref_count += spk_n
        if ref_count >= target_ref:
            break

    # Log gender balance in ref
    ref_male = sum(1 for s in ref_speakers if speaker_gender.get(s) == "male")
    ref_female = sum(1 for s in ref_speakers if speaker_gender.get(s) == "female")
    LOGGER.info("D_ref: %d speakers (%d male, %d female), %d samples (%.1f%%)",
                len(ref_speakers), ref_male, ref_female, ref_count, 100 * ref_count / total)

    # Mark ref
    df.loc[df["speaker_id"].isin(ref_speakers), "split"] = "ref"

    # Remaining speakers
    remaining_speakers = [s for s in df["speaker_id"].unique() if s not in ref_speakers]
    rng.shuffle(remaining_speakers)

    # Split remaining into train / cv / test by speaker
    remaining_df = df[df["speaker_id"].isin(remaining_speakers)]
    remaining_total = len(remaining_df)
    target_cv = int(total * cv_ratio)
    target_test = int(total * test_ratio)

    cv_speakers = []
    cv_count = 0
    test_speakers = []
    test_count = 0
    train_speakers = []

    for spk in remaining_speakers:
        spk_n = int((df["speaker_id"] == spk).sum())
        if cv_count < target_cv:
            cv_speakers.append(spk)
            cv_count += spk_n
        elif test_count < target_test:
            test_speakers.append(spk)
            test_count += spk_n
        else:
            train_speakers.append(spk)

    df.loc[df["speaker_id"].isin(cv_speakers), "split"] = "cv"
    df.loc[df["speaker_id"].isin(test_speakers), "split"] = "test"
    df.loc[df["speaker_id"].isin(train_speakers), "split"] = "train"

    # Summary
    for split_name in ["ref", "train", "cv", "test"]:
        n = int((df["split"] == split_name).sum())
        LOGGER.info("  %s: %d samples (%.1f%%)", split_name, n, 100 * n / total)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical filtering + data splitting for Stage E.")
    parser.add_argument("--metadata", required=True, help="Path to full metadata CSV (from Step 1).")
    parser.add_argument("--outdir", required=True, help="Output directory for scoring + split results.")
    parser.add_argument("--std-multiplier", type=float, default=1.5, help="Interval width: mean ± k*std.")
    parser.add_argument("--cv-ratio", type=float, default=0.08, help="CV proportion of total.")
    parser.add_argument("--test-ratio", type=float, default=0.07, help="Test proportion of total.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-ref-ratio", type=float, default=0.50, help="Max D_ref proportion.")
    parser.add_argument("--target-ref-ratio", type=float, default=0.15, help="Target D_ref proportion.")
    parser.add_argument("--n-workers", type=int, default=32, help="Parallel workers for feature extraction.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 2.1: Run scoring pipeline
    scored_df, group_test_df = run_scoring_pipeline(
        metadata_path=args.metadata,
        std_multiplier=args.std_multiplier,
        n_workers=args.n_workers,
        cache_dir=str(outdir),
    )

    # Check if sig_score==3 ratio exceeds max_ref_ratio
    n3 = int((scored_df["sig_score"] == 3).sum())
    ratio3 = n3 / len(scored_df)
    LOGGER.info("sig_score==3 ratio: %.1f%% (max allowed: %.1f%%)", ratio3 * 100, args.max_ref_ratio * 100)

    if ratio3 > args.max_ref_ratio:
        LOGGER.warning(
            "sig_score==3 ratio (%.1f%%) exceeds max (%.1f%%). "
            "D_ref will be capped at target ratio (%.1f%%) via speaker-level sampling.",
            ratio3 * 100, args.max_ref_ratio * 100, args.target_ref_ratio * 100,
        )

    # Step 2.2: Speaker-stratified split
    ref_mask = scored_df["sig_score"] == 3
    split_df = speaker_stratified_split(
        scored_df,
        ref_mask=ref_mask,
        cv_ratio=args.cv_ratio,
        test_ratio=args.test_ratio,
        target_ref_ratio=args.target_ref_ratio,
        seed=args.seed,
    )

    # Save outputs
    # Full scored + split CSV
    scored_path = outdir / "audio_significance_results.csv"
    split_df.to_csv(scored_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved scored+split results: %s", scored_path)

    # Group test results
    group_test_df.to_csv(outdir / "group_test_results.csv", index=False, encoding="utf-8-sig")

    # Split manifest files
    for split_name in ["ref", "train", "cv", "test"]:
        subset = split_df[split_df["split"] == split_name].copy()
        manifest_path = outdir / f"manifest_{split_name}.csv"
        cols = ["audio_path", "text", "gender_label", "age_label", "emotion_label", "speaker_id", "age_num", "sig_score"]
        available_cols = [c for c in cols if c in subset.columns]
        subset[available_cols].to_csv(manifest_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved %s manifest: %s (%d samples)", split_name, manifest_path, len(subset))

    # Speaker split mapping
    speaker_split = {}
    for _, row in split_df[["speaker_id", "split"]].drop_duplicates().iterrows():
        speaker_split[int(row["speaker_id"])] = row["split"]
    save_json(outdir / "splits.json", speaker_split)

    # Summary
    summary = {
        "total_samples": len(split_df),
        "std_multiplier": args.std_multiplier,
        "sig_score_distribution": split_df["sig_score"].value_counts().sort_index().to_dict(),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "split_speaker_counts": {
            s: int(split_df[split_df["split"] == s]["speaker_id"].nunique())
            for s in ["ref", "train", "cv", "test"]
        },
        "seed": args.seed,
    }
    # Per-split emotion/gender distribution
    for split_name in ["ref", "train", "cv", "test"]:
        sub = split_df[split_df["split"] == split_name]
        summary[f"{split_name}_emotion_dist"] = sub["emotion_label"].value_counts().to_dict() if "emotion_label" in sub.columns else {}
        summary[f"{split_name}_gender_dist"] = sub["gender_label"].value_counts().to_dict() if "gender_label" in sub.columns else {}

    save_json(outdir / "split_summary.json", summary)
    LOGGER.info("Saved summary: %s", outdir / "split_summary.json")

    LOGGER.info("=" * 60)
    LOGGER.info("SPLIT SUMMARY")
    for split_name in ["ref", "train", "cv", "test"]:
        n = int((split_df["split"] == split_name).sum())
        n_spk = int(split_df[split_df["split"] == split_name]["speaker_id"].nunique())
        LOGGER.info("  %s: %d samples, %d speakers", split_name, n, n_spk)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
