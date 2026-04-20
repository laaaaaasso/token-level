"""Step 2 (adapted): Statistical filtering + data splitting for multi-dataset.

Adapted from split_data.py for HiFi-TTS data without gender/age/emotion labels.
Uses 3 acoustic-feature-based quality dimensions instead:
  1. Speaker F0 consistency: sample F0 within speaker's typical range
  2. Source quality: energy/spectral features within dataset_source's typical range
  3. Prosodic typicality: prosodic features within speaker's typical range

Scoring: sig_score = sum of 3 dimensions (0-3), same structure as original.
Settings: std_multiplier=1.5, sig_score==3 for D_ref, target <= 50%.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction import FEATURE_COLUMNS, extract_single_audio_features
from utils import save_json

LOGGER = logging.getLogger("split_data_multi")


# ── Feature extraction (parallel) ──────────────────────────────────────

def _extract_one(args):
    """Worker: extract features for a single audio file."""
    row_id, audio_path, base_dir = args
    full_path = Path(base_dir) / audio_path
    if not full_path.exists():
        empty = {k: np.nan for k in FEATURE_COLUMNS}
        empty["feature_status"] = "missing"
        return row_id, empty
    try:
        feats = extract_single_audio_features(str(full_path))
        feats["feature_status"] = "ok"
        return row_id, feats
    except Exception as e:
        empty = {k: np.nan for k in FEATURE_COLUMNS}
        empty["feature_status"] = "error"
        return row_id, empty


def extract_features_parallel(metadata_df, base_dir, n_workers=32, cache_path=None):
    """Extract audio features with multiprocessing and optional caching."""
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

    work_items = []
    for idx, row in metadata_df.iterrows():
        work_items.append((idx, row["audio_path"], base_dir))

    LOGGER.info("Extracting features for %d samples using %d workers...",
                len(work_items), n_workers)
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
                elapsed, elapsed / max(len(work_items), 1))

    rows = [results[idx] for idx in metadata_df.index]
    feat_df = pd.DataFrame(rows, index=metadata_df.index)

    status_counts = feat_df["feature_status"].value_counts()
    LOGGER.info("Feature status: %s", status_counts.to_dict())

    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(cache, index=True)
        LOGGER.info("Saved feature cache: %s", cache)

    return feat_df


# ── Adapted 3-dimension scoring ────────────────────────────────────────

def _build_group_intervals(df, group_col, features, std_multiplier=1.5):
    """Build per-group mean ± k*std intervals for each feature."""
    cache = {}
    for label, grp in df.groupby(group_col, dropna=True):
        label_cache = {}
        for feat in features:
            vals = pd.to_numeric(grp[feat], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 5:
                label_cache[feat] = {"lower": np.nan, "upper": np.nan, "n": 0}
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            label_cache[feat] = {
                "mean": mean,
                "std": std,
                "lower": mean - std_multiplier * std,
                "upper": mean + std_multiplier * std,
                "n": int(vals.size),
            }
        cache[str(label)] = label_cache
    return cache


def _in_interval(value, interval):
    """Check if value falls in [lower, upper]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v):
        return False
    low = interval.get("lower", np.nan)
    high = interval.get("upper", np.nan)
    if not (np.isfinite(low) and np.isfinite(high)):
        return False
    return bool(low <= v <= high)


def score_dimension_1(df, std_multiplier=1.5):
    """Dim 1 - Speaker F0 consistency.

    For each sample, check if its F0 features are within its speaker's
    typical range (mean ± k*std). Passes if both f0_mean and f0_std
    are within range.
    """
    features = ["f0_mean", "f0_std"]
    intervals = _build_group_intervals(df, "speaker_id", features, std_multiplier)

    scores = []
    for _, row in df.iterrows():
        spk = str(row["speaker_id"])
        spk_intervals = intervals.get(spk, {})
        passed = 0
        for feat in features:
            if _in_interval(row.get(feat), spk_intervals.get(feat, {})):
                passed += 1
        # Pass if both features are within range
        scores.append(1 if passed >= 2 else 0)

    LOGGER.info("Dim1 (Speaker F0): %d / %d passed (%.1f%%)",
                sum(scores), len(scores), 100 * sum(scores) / max(len(scores), 1))
    return scores, intervals


def score_dimension_2(df, std_multiplier=1.5):
    """Dim 2 - Source quality (energy/spectral).

    For each sample, check if its energy and spectral features are within
    its dataset_source's typical range. Passes if >= 2 of 3 features pass.
    """
    features = ["energy_mean", "energy_std", "spec_centroid_mean"]
    intervals = _build_group_intervals(df, "dataset_source", features, std_multiplier)

    scores = []
    for _, row in df.iterrows():
        src = str(row["dataset_source"])
        src_intervals = intervals.get(src, {})
        passed = 0
        for feat in features:
            if _in_interval(row.get(feat), src_intervals.get(feat, {})):
                passed += 1
        scores.append(1 if passed >= 2 else 0)

    LOGGER.info("Dim2 (Source quality): %d / %d passed (%.1f%%)",
                sum(scores), len(scores), 100 * sum(scores) / max(len(scores), 1))
    return scores, intervals


def score_dimension_3(df, std_multiplier=1.5):
    """Dim 3 - Prosodic typicality.

    For each sample, check if its prosodic/dynamic features are within
    its speaker's typical range. Passes if >= 2 of 4 features pass.
    """
    features = ["f0_range", "energy_range", "spec_flux_mean", "mfcc_delta_std"]
    intervals = _build_group_intervals(df, "speaker_id", features, std_multiplier)

    scores = []
    for _, row in df.iterrows():
        spk = str(row["speaker_id"])
        spk_intervals = intervals.get(spk, {})
        passed = 0
        for feat in features:
            if _in_interval(row.get(feat), spk_intervals.get(feat, {})):
                passed += 1
        scores.append(1 if passed >= 2 else 0)

    LOGGER.info("Dim3 (Prosodic typicality): %d / %d passed (%.1f%%)",
                sum(scores), len(scores), 100 * sum(scores) / max(len(scores), 1))
    return scores, intervals


def run_scoring(df, std_multiplier=1.5):
    """Run all 3 dimensions of scoring. Returns df with sig_score column."""
    out = df.copy()

    LOGGER.info("Running 3-dimension scoring (std_multiplier=%.2f)...", std_multiplier)

    dim1_scores, dim1_intervals = score_dimension_1(out, std_multiplier)
    dim2_scores, dim2_intervals = score_dimension_2(out, std_multiplier)
    dim3_scores, dim3_intervals = score_dimension_3(out, std_multiplier)

    out["dim1_speaker_f0"] = dim1_scores
    out["dim2_source_quality"] = dim2_scores
    out["dim3_prosodic"] = dim3_scores
    out["sig_score"] = out["dim1_speaker_f0"] + out["dim2_source_quality"] + out["dim3_prosodic"]

    dist = out["sig_score"].value_counts().sort_index()
    LOGGER.info("sig_score distribution:\n%s", dist.to_string())
    n3 = int((out["sig_score"] == 3).sum())
    LOGGER.info("sig_score == 3: %d / %d (%.1f%%)", n3, len(out), 100 * n3 / len(out))

    stats_cache = {
        "dim1_speaker_f0": dim1_intervals,
        "dim2_source_quality": dim2_intervals,
        "dim3_prosodic": dim3_intervals,
    }
    return out, stats_cache


# ── Data splitting ─────────────────────────────────────────────────────

def speaker_stratified_split(df, target_ref_ratio=0.15, cv_ratio=0.08,
                              test_ratio=0.07, seed=42):
    """Split by speaker_id into D_ref / D_train / CV / D_test.

    D_ref: speakers with highest sig_score==3 rate, up to target_ref_ratio.
    CV and D_test drawn from remaining speakers.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["split"] = "unknown"
    total = len(df)

    # Speaker-level sig_score==3 rate
    speaker_ref_rate = df.groupby("speaker_id")["sig_score"].apply(
        lambda s: (s == 3).mean()
    ).sort_values(ascending=False)

    LOGGER.info("Speaker sig_score==3 rates:")
    for spk, rate in speaker_ref_rate.items():
        n = int((df["speaker_id"] == spk).sum())
        LOGGER.info("  %s: %.1f%% (%d samples)", spk, rate * 100, n)

    # Sort speakers by ref rate, pick greedily until target
    all_speakers = speaker_ref_rate.index.tolist()
    target_ref = int(total * target_ref_ratio)

    ref_speakers = []
    ref_count = 0
    for spk in all_speakers:
        spk_n = int((df["speaker_id"] == spk).sum())
        if ref_count + spk_n <= int(target_ref * 1.2):  # 20% overshoot tolerance
            ref_speakers.append(spk)
            ref_count += spk_n
        if ref_count >= target_ref:
            break

    df.loc[df["speaker_id"].isin(ref_speakers), "split"] = "ref"
    LOGGER.info("D_ref: %d speakers, %d samples (%.1f%%)",
                len(ref_speakers), ref_count, 100 * ref_count / total)

    # Remaining speakers: split into train/cv/test
    remaining = [s for s in all_speakers if s not in ref_speakers]
    rng.shuffle(remaining)

    target_cv = int(total * cv_ratio)
    target_test = int(total * test_ratio)

    cv_speakers, cv_count = [], 0
    test_speakers, test_count = [], 0
    train_speakers = []

    for spk in remaining:
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

    for s in ["ref", "train", "cv", "test"]:
        n = int((df["split"] == s).sum())
        n_spk = int(df[df["split"] == s]["speaker_id"].nunique())
        LOGGER.info("  %s: %d samples (%.1f%%), %d speakers", s, n, 100 * n / total, n_spk)

    return df


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Statistical filtering + splitting for multi-dataset")
    parser.add_argument("--metadata", required=True,
                        help="Path to metadata.csv (from Step 1)")
    parser.add_argument("--data-dir", required=True,
                        help="Base directory containing audios/")
    parser.add_argument("--outdir", required=True,
                        help="Output directory")
    parser.add_argument("--std-multiplier", type=float, default=1.5)
    parser.add_argument("--target-ref-ratio", type=float, default=0.15)
    parser.add_argument("--cv-ratio", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=32)
    parser.add_argument("--features-cache", type=str, default=None,
                        help="Path to pre-extracted features_cache.parquet (skip extraction)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(args.metadata)
    LOGGER.info("Loaded metadata: %d rows, columns: %s", len(df), list(df.columns))

    # Step 1: Extract features (or load from pre-extracted cache)
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 1: Feature extraction")
    LOGGER.info("=" * 60)

    if args.features_cache and Path(args.features_cache).exists():
        LOGGER.info("Loading pre-extracted features from %s", args.features_cache)
        feat_df = pd.read_parquet(args.features_cache)
        LOGGER.info("Loaded %d rows of features", len(feat_df))
        if len(feat_df) != len(df):
            LOGGER.warning("Feature cache size (%d) != metadata size (%d)!",
                          len(feat_df), len(df))
    else:
        cache_path = outdir / "features_cache.parquet"
        feat_df = extract_features_parallel(
            df, base_dir=args.data_dir,
            n_workers=args.n_workers, cache_path=str(cache_path)
        )

    # Merge metadata + features (drop overlapping columns from features)
    overlap_cols = [c for c in feat_df.columns if c in df.columns]
    if overlap_cols:
        LOGGER.info("Dropping overlapping columns from features: %s", overlap_cols)
        feat_df_clean = feat_df.drop(columns=overlap_cols)
    else:
        feat_df_clean = feat_df
    full_df = pd.concat([df.reset_index(drop=True),
                         feat_df_clean.reset_index(drop=True)], axis=1)

    # Save features CSV
    feat_csv = outdir / "features.csv"
    feat_df.reset_index(drop=True).to_csv(feat_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved features: %s", feat_csv)

    # Step 2: Scoring
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 2: 3-dimension scoring (std_multiplier=%.2f)", args.std_multiplier)
    LOGGER.info("=" * 60)
    scored_df, stats_cache = run_scoring(full_df, std_multiplier=args.std_multiplier)

    # Check sig_score==3 ratio
    n3 = int((scored_df["sig_score"] == 3).sum())
    ratio3 = n3 / len(scored_df)
    LOGGER.info("sig_score==3: %d / %d (%.1f%%)", n3, len(scored_df), ratio3 * 100)

    if ratio3 > 0.50:
        LOGGER.warning("sig_score==3 ratio (%.1f%%) exceeds 50%%. "
                       "D_ref will be capped via speaker-level sampling.", ratio3 * 100)

    # Step 3: Split
    LOGGER.info("=" * 60)
    LOGGER.info("Phase 3: Speaker-stratified split")
    LOGGER.info("=" * 60)
    split_df = speaker_stratified_split(
        scored_df,
        target_ref_ratio=args.target_ref_ratio,
        cv_ratio=args.cv_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Save outputs
    scored_path = outdir / "audio_significance_results.csv"
    split_df.to_csv(scored_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved scored+split: %s", scored_path)

    # Per-split manifests
    for split_name in ["ref", "train", "cv", "test"]:
        subset = split_df[split_df["split"] == split_name].copy()
        cols = ["audio_path", "text", "speaker_id", "dataset_source",
                "duration_sec", "sig_score"]
        available = [c for c in cols if c in subset.columns]
        manifest_path = outdir / ("manifest_%s.csv" % split_name)
        subset[available].to_csv(manifest_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved %s manifest: %s (%d samples)", split_name, manifest_path, len(subset))

    # Speaker split mapping
    speaker_split = {}
    for _, row in split_df[["speaker_id", "split"]].drop_duplicates().iterrows():
        speaker_split[str(row["speaker_id"])] = row["split"]
    save_json(outdir / "splits.json", speaker_split)

    # Stats cache
    save_json(outdir / "stats_cache.json", {
        k: {
            spk: {feat: {kk: (float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv)
                         for kk, vv in fdict.items()}
                  for feat, fdict in spk_dict.items()}
            for spk, spk_dict in v.items()
        }
        for k, v in stats_cache.items()
    })

    # Summary
    summary = {
        "total_samples": len(split_df),
        "std_multiplier": args.std_multiplier,
        "sig_score_distribution": {int(k): int(v) for k, v in
                                    split_df["sig_score"].value_counts().sort_index().items()},
        "split_counts": {str(k): int(v) for k, v in split_df["split"].value_counts().to_dict().items()},
        "split_speaker_counts": {
            s: int(split_df[split_df["split"] == s]["speaker_id"].nunique())
            for s in ["ref", "train", "cv", "test"]
        },
        "per_split_dataset_source": {},
        "seed": args.seed,
    }
    for s in ["ref", "train", "cv", "test"]:
        sub = split_df[split_df["split"] == s]
        summary["per_split_dataset_source"][s] = {
            str(k): int(v) for k, v in sub["dataset_source"].value_counts().to_dict().items()
        }
        dur = float(sub["duration_sec"].sum()) / 3600
        summary["%s_hours" % s] = round(float(dur), 2)

    save_json(outdir / "split_summary.json", summary)
    LOGGER.info("Saved summary: %s", outdir / "split_summary.json")

    LOGGER.info("=" * 60)
    LOGGER.info("SPLIT SUMMARY")
    for s in ["ref", "train", "cv", "test"]:
        n = int((split_df["split"] == s).sum())
        n_spk = int(split_df[split_df["split"] == s]["speaker_id"].nunique())
        hrs = split_df[split_df["split"] == s]["duration_sec"].sum() / 3600
        LOGGER.info("  %s: %d samples, %d speakers, %.2f hours", s, n, n_spk, hrs)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
