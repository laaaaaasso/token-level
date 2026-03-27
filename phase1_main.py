"""CLI entry for Phase 1: train CosyVoice2 reference LM on D_ref.

The primary entry point is the shell script: scripts/phase1_train.sh
This Python entry point provides a simpler interface for individual steps.

Usage:
    # Prepare data only
    python phase1_main.py prepare --ref-manifest phase0_outputs/phase0_manifest_ref.jsonl \
        --audio-dir /data/zhenghao/data/crema_data_1000/audios

    # Export RM checkpoint
    python phase1_main.py export --model-dir phase1_outputs/checkpoints

    # Full pipeline (use shell script instead for better control)
    bash scripts/phase1_train.sh --run
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: reference model (RM) training on D_ref."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # prepare sub-command
    prep = subparsers.add_parser("prepare", help="Prepare parquet data from D_ref manifest")
    prep.add_argument("--ref-manifest", required=True)
    prep.add_argument("--audio-dir", required=True)
    prep.add_argument("--output-dir", default="phase1_outputs/data")
    prep.add_argument("--val-ratio", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--utts-per-shard", type=int, default=500)

    # export sub-command
    exp = subparsers.add_parser("export", help="Export frozen RM checkpoint")
    exp.add_argument("--model-dir", required=True)
    exp.add_argument("--output-dir", default="phase1_outputs/rm")

    args = parser.parse_args()

    if args.command == "prepare":
        from phase1.prepare_data import main as prepare_main
        sys.argv = [
            "prepare_data",
            "--ref-manifest", args.ref_manifest,
            "--audio-dir", args.audio_dir,
            "--output-dir", args.output_dir,
            "--val-ratio", str(args.val_ratio),
            "--seed", str(args.seed),
            "--utts-per-shard", str(args.utts_per_shard),
        ]
        prepare_main()

    elif args.command == "export":
        from phase1.export_checkpoint import main as export_main
        sys.argv = [
            "export_checkpoint",
            "--model-dir", args.model_dir,
            "--output-dir", args.output_dir,
        ]
        export_main()

    else:
        parser.print_help()
        print("\nFor full training pipeline, use: bash scripts/phase1_train.sh --run")


if __name__ == "__main__":
    main()
