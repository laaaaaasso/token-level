"""Main entry for audio significance validation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_extraction import FEATURE_COLUMNS, extract_features
from scoring import assign_significance_flags, rank_high_score_samples
from stats_validation import run_all_validations
from utils import ensure_outdir, load_age_mapping, load_metadata, save_json, setup_logging

LOGGER = logging.getLogger(__name__)


def _boxplot_or_placeholder(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Save a single boxplot; fallback to placeholder when data is unavailable."""
    plt.figure(figsize=(8, 5))
    valid = df[[group_col, value_col]].dropna()
    if valid.empty:
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
        plt.title(title)
        plt.axis("off")
    else:
        labels = sorted(valid[group_col].unique().tolist())
        data = [valid.loc[valid[group_col] == label, value_col].to_numpy() for label in labels]
        try:
            plt.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            plt.boxplot(data, labels=labels, showfliers=False)
        plt.title(title)
        plt.xlabel(group_col)
        plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _emotion_multi_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    """Save a 2x2 emotion-feature boxplot figure."""
    feats = ["f0_range", "energy_range", "spec_flux_mean", "mfcc_delta_std"]
    labels = sorted(df["emotion_label_norm"].dropna().unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]
        valid = df[["emotion_label_norm", feat]].dropna()
        if valid.empty or not labels:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(feat)
            continue
        data = [valid.loc[valid["emotion_label_norm"] == label, feat].to_numpy() for label in labels]
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Emotion One-vs-Rest Related Features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _sig_score_hist(df: pd.DataFrame, out_path: Path) -> None:
    """Save significance score histogram."""
    plt.figure(figsize=(7, 5))
    vals = pd.to_numeric(df["sig_score"], errors="coerce").dropna().to_numpy(dtype=float)
    bins = np.arange(-0.5, 4.5, 1.0)
    if vals.size:
        plt.hist(vals, bins=bins, rwidth=0.85)
    else:
        plt.text(0.5, 0.5, "No valid score", ha="center", va="center")
        plt.axis("off")
    plt.xticks([0, 1, 2, 3])
    plt.xlabel("sig_score")
    plt.ylabel("count")
    plt.title("Significance Score Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results(
    outdir: Path,
    metadata_df: pd.DataFrame,
    features_df: pd.DataFrame,
    group_test_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    high_df: pd.DataFrame,
    interval_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Save CSV, JSON and PNG outputs required by the task."""
    features_path = outdir / "features.csv"
    group_test_path = outdir / "group_test_results.csv"
    all_results_path = outdir / "audio_significance_results.csv"
    high_results_path = outdir / "high_score_audios.csv"
    summary_path = outdir / "summary.json"

    features_df.to_csv(features_path, index=False, encoding="utf-8-sig")
    group_test_df.to_csv(group_test_path, index=False, encoding="utf-8-sig")
    ranked_df.to_csv(all_results_path, index=False, encoding="utf-8-sig")
    high_df.to_csv(high_results_path, index=False, encoding="utf-8-sig")

    sig_dist = (
        ranked_df["sig_score"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    sig_dist = {str(k): int(v) for k, v in sig_dist.items()}

    task_sig_features = {}
    for task in ("gender", "age", "emotion"):
        task_df = group_test_df[group_test_df["task"] == task]
        task_sig_features[task] = int(task_df["significant"].sum())

    summary = {
        "total_samples": int(len(metadata_df)),
        "feature_status_counts": features_df["feature_status"].value_counts(dropna=False).to_dict(),
        "task_significant_feature_count": task_sig_features,
        "sig_score_distribution": sig_dist,
        "sig_score_eq_3_count": int((ranked_df["sig_score"] == 3).sum()),
        "sig_score_ge_2_count": int((ranked_df["sig_score"] >= 2).sum()),
        "high_score_threshold": 2,
        "interval_method": interval_config["interval_method"],
        "std_multiplier": interval_config["std_multiplier"],
        "iqr_multiplier": interval_config["iqr_multiplier"],
    }
    save_json(summary_path, summary)

    plot_df = ranked_df.copy()
    _boxplot_or_placeholder(
        plot_df[plot_df["gender_label_norm"].isin(["male", "female"])],
        group_col="gender_label_norm",
        value_col="f0_mean",
        title="Gender vs f0_mean",
        out_path=outdir / "gender_f0_boxplot.png",
    )

    age_plot_df = plot_df[plot_df["age_label_bin"].isin(["young", "old"])].copy()
    _boxplot_or_placeholder(
        age_plot_df,
        group_col="age_label_bin",
        value_col="f0_mean",
        title="Age Group vs f0_mean",
        out_path=outdir / "age_feature_boxplot.png",
    )

    _emotion_multi_boxplot(
        plot_df[plot_df["emotion_label_norm"].notna()].copy(),
        out_path=outdir / "emotion_feature_boxplot.png",
    )

    _sig_score_hist(plot_df, out_path=outdir / "sig_score_hist.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(
        description="Statistical significance tagging for gender/age/emotion from audio features."
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio load sample rate.")
    parser.add_argument("--top-db", type=float, default=30.0, help="Silence threshold parameter.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--interval-method",
        type=str,
        default="std",
        choices=["std", "iqr"],
        help="Group interval method used by per-sample scoring.",
    )
    parser.add_argument(
        "--std-multiplier",
        type=float,
        default=1.5,
        help="Interval width for std method: mean ± k*std.",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="Interval width for iqr method: [Q1-k*IQR, Q3+k*IQR].",
    )
    parser.add_argument("--age-map-json", default=None, help="Optional custom age mapping JSON file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> None:
    """Run complete pipeline end-to-end."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    outdir = ensure_outdir(args.outdir)
    LOGGER.info("Loading metadata from %s", args.metadata)

    age_map = load_age_mapping(args.age_map_json)
    metadata_df = load_metadata(args.metadata, age_map)
    LOGGER.info("Metadata loaded: %s rows", len(metadata_df))

    LOGGER.info("Extracting acoustic features...")
    feature_df = extract_features(metadata_df, sample_rate=args.sample_rate, top_db=args.top_db)
    feature_export = feature_df.reset_index(drop=True)

    full_df = pd.concat([metadata_df.reset_index(drop=True), feature_export], axis=1)
    LOGGER.info("Running group-level statistical validations...")
    interval_config = {
        "interval_method": args.interval_method,
        "std_multiplier": args.std_multiplier,
        "iqr_multiplier": args.iqr_multiplier,
    }
    group_test_df, stats_cache = run_all_validations(
        full_df,
        alpha=args.alpha,
        interval_method=args.interval_method,
        std_multiplier=args.std_multiplier,
        iqr_multiplier=args.iqr_multiplier,
    )

    LOGGER.info("Assigning per-sample significance flags...")
    scored_df = assign_significance_flags(full_df, stats_cache)
    ranked_df, high_df = rank_high_score_samples(scored_df, high_score_threshold=2)

    summary = save_results(
        outdir=outdir,
        metadata_df=metadata_df,
        features_df=feature_export,
        group_test_df=group_test_df,
        ranked_df=ranked_df,
        high_df=high_df,
        interval_config=interval_config,
    )

    LOGGER.info("Pipeline done. Outputs saved to %s", outdir)
    LOGGER.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
