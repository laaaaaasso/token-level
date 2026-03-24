"""CLI entry for Phase 0 data preparation."""

from __future__ import annotations

import argparse
import json
import logging

from phase0 import run_phase0_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser."""
    parser = argparse.ArgumentParser(description="Phase 0: CosyVoice2 + RHO-1 data preparation and layering.")
    parser.add_argument("--train-manifest", required=True, help="Path to D_train manifest (.csv or .jsonl).")
    parser.add_argument("--ref-manifest", default=None, help="Path to D_ref manifest (.csv or .jsonl).")
    parser.add_argument(
        "--ref-flag-column",
        default=None,
        help="Optional column in D_train manifest that marks D_ref membership.",
    )
    parser.add_argument("--outdir", required=True, help="Output directory for phase0 manifests.")
    parser.add_argument(
        "--audio-base-dir",
        default=None,
        help="Optional base directory for relative audio paths.",
    )
    parser.add_argument("--speech-sample-rate", type=int, default=16000, help="Sample rate used by speech tokenizer.")
    parser.add_argument("--speech-vocab-size", type=int, default=1024, help="Token vocab size for speech tokenizer.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser


def main() -> None:
    """Run Phase 0 pipeline via CLI."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = run_phase0_pipeline(
        train_manifest=args.train_manifest,
        ref_manifest=args.ref_manifest,
        ref_flag_column=args.ref_flag_column,
        outdir=args.outdir,
        audio_base_dir=args.audio_base_dir,
        speech_sample_rate=args.speech_sample_rate,
        speech_vocab_size=args.speech_vocab_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
