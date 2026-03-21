"""Per-sample significance flag assignment and ranking."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _in_interval(value: Any, interval: Dict[str, float]) -> bool:
    """Check if value falls in [lower, upper] interval."""
    try:
        v = float(value)
    except Exception:
        return False
    if not np.isfinite(v):
        return False
    low = interval.get("lower", np.nan)
    high = interval.get("upper", np.nan)
    if not (np.isfinite(low) and np.isfinite(high)):
        return False
    return bool(low <= v <= high)


def assign_gender_sig(row: pd.Series, stats_cache: Dict[str, Any], feature_df: pd.DataFrame | None = None) -> int:
    """Assign gender significance flag for one row."""
    del feature_df  # kept for API compatibility
    gender_cache = stats_cache.get("gender", {})
    tests = gender_cache.get("tests", {})
    f0_test = tests.get("f0_mean", {})
    if int(f0_test.get("significant", 0)) != 1:
        return 0

    label = row.get("gender_label_norm")
    val = row.get("f0_mean")
    interval = gender_cache.get("intervals", {}).get(label, {}).get("f0_mean", {})
    return 1 if _in_interval(val, interval) else 0


def assign_age_sig(row: pd.Series, stats_cache: Dict[str, Any], feature_df: pd.DataFrame | None = None) -> int:
    """Assign age significance flag for one row."""
    del feature_df  # kept for API compatibility
    age_cache = stats_cache.get("age", {})
    tests = age_cache.get("tests", {})
    label = row.get("age_label_bin")
    intervals = age_cache.get("intervals", {}).get(label, {})

    votes = 0
    for feat in ("f0_mean", "pause_ratio"):
        test = tests.get(feat, {})
        if int(test.get("significant", 0)) != 1:
            continue
        if _in_interval(row.get(feat), intervals.get(feat, {})):
            votes += 1
    return 1 if votes >= 1 else 0


def assign_emotion_sig(row: pd.Series, stats_cache: Dict[str, Any], feature_df: pd.DataFrame | None = None) -> int:
    """Assign emotion significance flag for one row."""
    del feature_df  # kept for API compatibility
    emotion_cache = stats_cache.get("emotion", {})
    label = row.get("emotion_label_norm")
    per_emo = emotion_cache.get("per_emotion", {}).get(label, {})
    tests = per_emo.get("tests", {})
    intervals = per_emo.get("intervals", {})

    matched = 0
    for feat in ("f0_range", "energy_range", "spec_flux_mean", "mfcc_delta_std"):
        test = tests.get(feat, {})
        if int(test.get("significant", 0)) != 1:
            continue
        if _in_interval(row.get(feat), intervals.get(feat, {})):
            matched += 1
    return 1 if matched >= 2 else 0


def _count_age_votes(row: pd.Series, stats_cache: Dict[str, Any]) -> int:
    age_cache = stats_cache.get("age", {})
    tests = age_cache.get("tests", {})
    label = row.get("age_label_bin")
    intervals = age_cache.get("intervals", {}).get(label, {})
    votes = 0
    for feat in ("f0_mean", "pause_ratio"):
        test = tests.get(feat, {})
        if int(test.get("significant", 0)) == 1 and _in_interval(row.get(feat), intervals.get(feat, {})):
            votes += 1
    return votes


def _count_emotion_matches(row: pd.Series, stats_cache: Dict[str, Any]) -> int:
    emotion_cache = stats_cache.get("emotion", {})
    label = row.get("emotion_label_norm")
    per_emo = emotion_cache.get("per_emotion", {}).get(label, {})
    tests = per_emo.get("tests", {})
    intervals = per_emo.get("intervals", {})
    matched = 0
    for feat in ("f0_range", "energy_range", "spec_flux_mean", "mfcc_delta_std"):
        test = tests.get(feat, {})
        if int(test.get("significant", 0)) == 1 and _in_interval(row.get(feat), intervals.get(feat, {})):
            matched += 1
    return matched


def assign_significance_flags(feature_df: pd.DataFrame, stats_cache: Dict[str, Any]) -> pd.DataFrame:
    """Assign gender_sig, age_sig, emotion_sig and total sig_score."""
    out = feature_df.copy()
    out["gender_sig"] = out.apply(assign_gender_sig, axis=1, args=(stats_cache, out))
    out["age_sig"] = out.apply(assign_age_sig, axis=1, args=(stats_cache, out))
    out["emotion_sig"] = out.apply(assign_emotion_sig, axis=1, args=(stats_cache, out))
    out["age_vote_count"] = out.apply(_count_age_votes, axis=1, args=(stats_cache,))
    out["emotion_match_count"] = out.apply(_count_emotion_matches, axis=1, args=(stats_cache,))
    out["sig_score"] = out["gender_sig"] + out["age_sig"] + out["emotion_sig"]
    return out


def rank_high_score_samples(
    scored_df: pd.DataFrame,
    high_score_threshold: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort all samples by score and return high-score subset."""
    sort_cols = ["sig_score", "emotion_match_count", "age_vote_count", "duration_sec"]
    available_cols = [c for c in sort_cols if c in scored_df.columns]
    ranked = scored_df.sort_values(by=available_cols, ascending=[False] * len(available_cols)).reset_index(drop=True)
    high = ranked[ranked["sig_score"] >= high_score_threshold].reset_index(drop=True)
    return ranked, high
