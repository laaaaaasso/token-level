"""CLI entry for Phase 1: train CosyVoice2 reference LM on D_ref."""

from __future__ import annotations

import argparse
import json

from phase1 import run_phase1_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1: reference model (RM) training on D_ref.")
    parser.add_argument("--ref-manifest", default="phase0_outputs/phase0_manifest_ref.jsonl")
    parser.add_argument("--output-dir", default="phase1_outputs")
    parser.add_argument("--cosyvoice-root", default="CosyVoice")
    parser.add_argument(
        "--base-config",
        default="CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml",
        help="Base CosyVoice2 YAML used to create Phase1 config.",
    )
    parser.add_argument(
        "--qwen-pretrain-path",
        default="",
        help="Path to CosyVoice-BlankEN (required when --run-training).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional initialization checkpoint, usually CosyVoice2 llm.pt.",
    )
    parser.add_argument(
        "--onnx-path",
        default=".",
        help="Passed to CosyVoice train.py to avoid None env issue.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-epoch", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-per-step", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-utts-per-parquet", type=int, default=500)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Actually launch torchrun training. If omitted, only prepare data/config/command.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_phase1_training(
        ref_manifest=args.ref_manifest,
        output_dir=args.output_dir,
        cosyvoice_root=args.cosyvoice_root,
        base_config=args.base_config,
        qwen_pretrain_path=args.qwen_pretrain_path,
        init_checkpoint=args.init_checkpoint,
        onnx_path=args.onnx_path,
        learning_rate=args.learning_rate,
        max_epoch=args.max_epoch,
        val_ratio=args.val_ratio,
        log_interval=args.log_interval,
        save_per_step=args.save_per_step,
        seed=args.seed,
        max_samples=args.max_samples,
        num_utts_per_parquet=args.num_utts_per_parquet,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        prefetch=args.prefetch,
        pin_memory=args.pin_memory,
        use_amp=args.use_amp,
        run_training=args.run_training,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
