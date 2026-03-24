"""Group-level statistical validation for labels vs acoustic features."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from utils import safe_numeric

LOGGER = logging.getLogger(__name__)


def run_two_group_test(
    group_a: Iterable[Any],
    group_b: Iterable[Any],
    alpha: float = 0.05,
    min_samples: int = 5,
    shapiro_max_n: int = 5000,
) -> Dict[str, Any]:
    """Run t-test or Mann-Whitney U automatically with normality pre-check."""
    a = safe_numeric(group_a)
    b = safe_numeric(group_b)
    if len(a) < min_samples or len(b) < min_samples:
        return {
            "test_name": "insufficient_samples",
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": 0,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
        }

    use_ttest = False
    zero_var = np.ptp(a) == 0 or np.ptp(b) == 0
    if zero_var:
        use_ttest = False
    elif len(a) <= shapiro_max_n and len(b) <= shapiro_max_n:
        try:
            p_norm_a = stats.shapiro(a).pvalue
            p_norm_b = stats.shapiro(b).pvalue
            use_ttest = (p_norm_a >= 0.05) and (p_norm_b >= 0.05)
        except Exception:
            use_ttest = False
    else:
        use_ttest = True

    if use_ttest:
        stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        test_name = "ttest_ind_welch"
    else:
        stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "mannwhitneyu"

    return {
        "test_name": test_name,
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": int(p_val < alpha),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def _build_group_intervals(
    df: pd.DataFrame,
    label_col: str,
    features: List[str],
    interval_method: str = "std",
    std_multiplier: float = 1.5,
    iqr_multiplier: float = 1.5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build group-specific intervals for each feature."""
    method = interval_method.lower().strip()
    if method not in {"std", "iqr"}:
        raise ValueError(f"Unsupported interval_method: {interval_method}")

    cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, grp in df.groupby(label_col, dropna=True):
        label_cache: Dict[str, Dict[str, float]] = {}
        for feat in features:
            vals = pd.to_numeric(grp[feat], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                label_cache[feat] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "n": 0.0,
                    "interval_method": method,
                }
                continue

            mean = float(np.mean(vals))
            std = float(np.std(vals))
            q1 = float(np.quantile(vals, 0.25))
            q3 = float(np.quantile(vals, 0.75))
            iqr = float(q3 - q1)
            if method == "std":
                lower = mean - std_multiplier * std
                upper = mean + std_multiplier * std
            else:
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr

            label_cache[feat] = {
                "mean": mean,
                "std": std,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": float(lower),
                "upper": float(upper),
                "n": float(vals.size),
                "interval_method": method,
            }
        cache[str(label)] = label_cache
    return cache


def validate_gender(
    feature_df: pd.DataFrame,
    alpha: float = 0.05,
    interval_method: str = "std",
    std_multiplier: float = 1.5,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate gender labels via two-group tests."""
    features = ["f0_mean", "spec_centroid_mean"]
    df = feature_df[feature_df["gender_label_norm"].isin(["male", "female"])].copy()

    rows: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {
        "primary_feature": "f0_mean",
        "features": features,
        "tests": {},
        "intervals": _build_group_intervals(
            df,
            "gender_label_norm",
            features,
            interval_method=interval_method,
            std_multiplier=std_multiplier,
            iqr_multiplier=iqr_multiplier,
        ),
        "interval_method": interval_method,
        "std_multiplier": std_multiplier,
        "iqr_multiplier": iqr_multiplier,
    }

    male = df[df["gender_label_norm"] == "male"]
    female = df[df["gender_label_norm"] == "female"]
    for feat in features:
        result = run_two_group_test(male[feat], female[feat], alpha=alpha)
        cache["tests"][feat] = result
        rows.append(
            {
                "task": "gender",
                "label_name": "male_vs_female",
                "feature": feat,
                "test_name": result["test_name"],
                "statistic": result["statistic"],
                "p_value": result["p_value"],
                "significant": result["significant"],
                "n_pos": result["n_a"],
                "n_neg": result["n_b"],
            }
        )
    return pd.DataFrame(rows), cache


def validate_age(
    feature_df: pd.DataFrame,
    alpha: float = 0.05,
    interval_method: str = "std",
    std_multiplier: float = 1.5,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate age labels (young vs old) via two-group tests."""
    features = ["f0_mean", "pause_ratio"]
    df = feature_df[feature_df["age_label_bin"].isin(["young", "old"])].copy()

    rows: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {
        "primary_features": features,
        "features": features,
        "tests": {},
        "intervals": _build_group_intervals(
            df,
            "age_label_bin",
            features,
            interval_method=interval_method,
            std_multiplier=std_multiplier,
            iqr_multiplier=iqr_multiplier,
        ),
        "interval_method": interval_method,
        "std_multiplier": std_multiplier,
        "iqr_multiplier": iqr_multiplier,
    }

    young = df[df["age_label_bin"] == "young"]
    old = df[df["age_label_bin"] == "old"]
    for feat in features:
        result = run_two_group_test(young[feat], old[feat], alpha=alpha)
        cache["tests"][feat] = result
        rows.append(
            {
                "task": "age",
                "label_name": "young_vs_old",
                "feature": feat,
                "test_name": result["test_name"],
                "statistic": result["statistic"],
                "p_value": result["p_value"],
                "significant": result["significant"],
                "n_pos": result["n_a"],
                "n_neg": result["n_b"],
            }
        )
    return pd.DataFrame(rows), cache


def validate_emotion(
    feature_df: pd.DataFrame,
    alpha: float = 0.05,
    interval_method: str = "std",
    std_multiplier: float = 1.5,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate emotion labels using one-vs-rest tests."""
    features = ["f0_range", "energy_range", "spec_flux_mean", "mfcc_delta_std"]
    valid_emotions = {"neutral", "happy", "sad", "angry", "fear", "surprised"}
    df = feature_df[feature_df["emotion_label_norm"].isin(valid_emotions)].copy()

    rows: List[Dict[str, Any]] = []
    intervals = _build_group_intervals(
        df,
        "emotion_label_norm",
        features,
        interval_method=interval_method,
        std_multiplier=std_multiplier,
        iqr_multiplier=iqr_multiplier,
    )
    per_emotion: Dict[str, Any] = {}

    for emo in sorted(df["emotion_label_norm"].dropna().unique().tolist()):
        pos = df[df["emotion_label_norm"] == emo]
        neg = df[df["emotion_label_norm"] != emo]
        tests: Dict[str, Any] = {}
        for feat in features:
            result = run_two_group_test(pos[feat], neg[feat], alpha=alpha)
            tests[feat] = result
            rows.append(
                {
                    "task": "emotion",
                    "label_name": f"{emo}_vs_rest",
                    "feature": feat,
                    "test_name": result["test_name"],
                    "statistic": result["statistic"],
                    "p_value": result["p_value"],
                    "significant": result["significant"],
                    "n_pos": result["n_a"],
                    "n_neg": result["n_b"],
                }
            )
        per_emotion[emo] = {"tests": tests, "intervals": intervals.get(emo, {})}

    cache: Dict[str, Any] = {
        "features": features,
        "per_emotion": per_emotion,
        "interval_method": interval_method,
        "std_multiplier": std_multiplier,
        "iqr_multiplier": iqr_multiplier,
    }
    return pd.DataFrame(rows), cache


def run_all_validations(
    feature_df: pd.DataFrame,
    alpha: float = 0.05,
    interval_method: str = "std",
    std_multiplier: float = 1.5,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run gender/age/emotion group-level validations."""
    gender_df, gender_cache = validate_gender(
        feature_df,
        alpha=alpha,
        interval_method=interval_method,
        std_multiplier=std_multiplier,
        iqr_multiplier=iqr_multiplier,
    )
    age_df, age_cache = validate_age(
        feature_df,
        alpha=alpha,
        interval_method=interval_method,
        std_multiplier=std_multiplier,
        iqr_multiplier=iqr_multiplier,
    )
    emotion_df, emotion_cache = validate_emotion(
        feature_df,
        alpha=alpha,
        interval_method=interval_method,
        std_multiplier=std_multiplier,
        iqr_multiplier=iqr_multiplier,
    )
    group_test_results = pd.concat([gender_df, age_df, emotion_df], ignore_index=True)
    stats_cache = {
        "gender": gender_cache,
        "age": age_cache,
        "emotion": emotion_cache,
    }
    LOGGER.info("Validation finished. Total tests=%s", len(group_test_results))
    return group_test_results, stats_cache
