#!/usr/bin/env python3
"""Phase 1: export frozen RM checkpoint from training output.

Scans the model_dir for epoch_*_whole.pt checkpoints, identifies best (lowest
CV loss) and last, copies them as rm_best.pt / rm_last.pt / rm_frozen.pt.

Usage:
    python -m phase1.export_checkpoint \
        --model-dir phase1_outputs/checkpoints \
        --output-dir phase1_outputs/rm
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_checkpoints(model_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find best and last checkpoint in model_dir.

    CosyVoice2 saves checkpoints as epoch_N_whole.pt with sidecar .yaml files.
    The .yaml contains loss_dict.loss from the CV run.
    """
    ckpts = sorted(model_dir.glob("epoch_*_whole.pt"))
    if not ckpts:
        # Also check step-level checkpoints
        ckpts = sorted(model_dir.glob("epoch_*_step_*.pt"))
    if not ckpts:
        logger.warning("No checkpoints found in %s", model_dir)
        return None, None

    last_ckpt = ckpts[-1]
    best_ckpt = None
    best_loss = float("inf")

    for ckpt in ckpts:
        sidecar = ckpt.with_suffix(".yaml")
        if not sidecar.exists():
            continue
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            loss = meta.get("loss_dict", {}).get("loss", None)
            if loss is not None and float(loss) < best_loss:
                best_loss = float(loss)
                best_ckpt = ckpt
        except Exception as e:
            logger.warning("Failed to parse %s: %s", sidecar, e)

    if best_ckpt is None:
        best_ckpt = last_ckpt
        logger.info("No valid loss info found, using last checkpoint as best")

    return best_ckpt, last_ckpt


def export(model_dir: Path, output_dir: Path) -> Dict[str, str]:
    """Export frozen RM checkpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt, last_ckpt = find_checkpoints(model_dir)

    if best_ckpt is None:
        logger.error("No checkpoints to export!")
        return {}

    result = {}
    for name, src in [("rm_best.pt", best_ckpt), ("rm_last.pt", last_ckpt)]:
        if src is not None:
            dst = output_dir / name
            shutil.copy2(src, dst)
            result[name] = str(dst.resolve())
            logger.info("Exported %s -> %s", src.name, dst)

    # rm_frozen.pt is always the best checkpoint (used by Phase 2)
    frozen_dst = output_dir / "rm_frozen.pt"
    shutil.copy2(best_ckpt, frozen_dst)
    result["rm_frozen.pt"] = str(frozen_dst.resolve())
    logger.info("Exported frozen RM: %s", frozen_dst)

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1: export frozen RM checkpoint")
    parser.add_argument("--model-dir", required=True,
                        help="Training checkpoint directory")
    parser.add_argument("--output-dir", default="phase1_outputs/rm",
                        help="Output directory for RM checkpoints")
    args = parser.parse_args()

    result = export(Path(args.model_dir), Path(args.output_dir))
    if result:
        summary_path = Path(args.output_dir) / "export_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, indent=2))
    else:
        print("ERROR: No checkpoints exported.")
        exit(1)


if __name__ == "__main__":
    main()
