#!/usr/bin/env python3
"""Phase 1 data preparation: convert D_ref manifest to CosyVoice2 parquet format.

Reads the Phase 0 output manifest (phase0_manifest_ref.jsonl), remaps audio paths
to the actual location on this server, loads raw audio bytes, and writes parquet
shards that CosyVoice2's data pipeline can directly consume.

The parquet contains: utt, audio_data, text.
Speech tokens and embeddings are extracted online during training (via onnx_path).

Usage:
    python -m phase1.prepare_data \
        --ref-manifest phase0_outputs/phase0_manifest_ref.jsonl \
        --audio-dir /data/zhenghao/data/crema_data_1000/audios \
        --output-dir phase1_outputs/data \
        --val-ratio 0.05
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_ref_manifest(path: Path) -> List[Dict[str, Any]]:
    """Read Phase 0 JSONL manifest."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Read %d rows from %s", len(rows), path)
    return rows


def resolve_audio_path(raw_path: str, audio_dir: Path) -> Path | None:
    """Remap manifest audio path to actual server path.

    Phase 0 may have recorded Windows or other-system paths.
    We extract just the filename and look it up in audio_dir.
    """
    # Handle Windows-style backslash paths on Linux
    filename = raw_path.replace("\\", "/").split("/")[-1]
    resolved = audio_dir / filename
    if resolved.exists():
        return resolved
    # Try original path as-is (for when paths are already correct)
    orig = Path(raw_path)
    if orig.exists():
        return orig
    return None


def validate_and_filter(
    rows: List[Dict[str, Any]],
    audio_dir: Path,
) -> List[Dict[str, Any]]:
    """Validate rows: ensure audio exists and text is non-empty."""
    valid = []
    skipped = 0
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            skipped += 1
            continue
        raw_audio = str(row.get("audio_path", ""))
        resolved = resolve_audio_path(raw_audio, audio_dir)
        if resolved is None:
            skipped += 1
            continue
        row["_resolved_audio"] = str(resolved)
        row["_utt"] = str(row.get("sample_id", Path(resolved).stem))
        valid.append(row)
    logger.info("Validated: %d valid, %d skipped", len(valid), skipped)
    return valid


def split_train_val(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into train and validation sets."""
    if val_ratio <= 0.0:
        return rows, []
    if len(rows) <= 1:
        return rows, rows[:1]
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_val = max(1, min(len(rows) - 1, int(round(len(rows) * val_ratio))))
    val_idx = set(indices[:n_val])
    train = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val = [rows[i] for i in range(len(rows)) if i in val_idx]
    logger.info("Split: %d train, %d val", len(train), len(val))
    return train, val


def write_parquet_shards(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str,
    utts_per_shard: int = 500,
) -> List[Path]:
    """Write rows to parquet shards, return list of shard paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_idx in range(0, len(rows), utts_per_shard):
        chunk = rows[shard_idx : shard_idx + utts_per_shard]
        records = []
        for row in chunk:
            audio_path = Path(row["_resolved_audio"])
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            records.append({
                "utt": row["_utt"],
                "audio_data": audio_bytes,
                "text": str(row.get("text", "")),
            })
        table = pa.Table.from_pylist(records)
        shard_path = output_dir / f"{prefix}_{shard_idx // utts_per_shard:05d}.parquet"
        pq.write_table(table, shard_path)
        shards.append(shard_path)
    logger.info("Wrote %d shards to %s (%d samples)", len(shards), output_dir, len(rows))
    return shards


def write_data_list(shards: List[Path], output_path: Path) -> None:
    """Write data.list file (one parquet path per line)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for p in shards:
            f.write(str(p.resolve()) + "\n")
    logger.info("Wrote data list: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: prepare D_ref parquet data")
    parser.add_argument("--ref-manifest", required=True,
                        help="Path to phase0_manifest_ref.jsonl")
    parser.add_argument("--audio-dir", required=True,
                        help="Directory containing actual audio files")
    parser.add_argument("--output-dir", default="phase1_outputs/data",
                        help="Output directory for parquets and data lists")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of D_ref to use as validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--utts-per-shard", type=int, default=500)
    args = parser.parse_args()

    ref_manifest = Path(args.ref_manifest).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read and validate
    rows = read_ref_manifest(ref_manifest)
    rows = validate_and_filter(rows, audio_dir)
    if not rows:
        raise RuntimeError("No valid samples after validation!")

    # 2. Split
    train_rows, val_rows = split_train_val(rows, args.val_ratio, args.seed)

    # 3. Write parquets
    train_shards = write_parquet_shards(
        train_rows, output_dir / "train", "train", args.utts_per_shard
    )
    if val_rows:
        val_shards = write_parquet_shards(
            val_rows, output_dir / "cv", "cv", args.utts_per_shard
        )
    else:
        val_shards = []

    # 4. Write data lists
    write_data_list(train_shards, output_dir / "train.data.list")
    if val_shards:
        write_data_list(val_shards, output_dir / "cv.data.list")

    # 5. Write summary
    summary = {
        "ref_manifest": str(ref_manifest),
        "audio_dir": str(audio_dir),
        "total_valid": len(rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "train_data_list": str((output_dir / "train.data.list").resolve()),
        "cv_data_list": str((output_dir / "cv.data.list").resolve()),
    }
    summary_path = output_dir / "prepare_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Done. Summary: %s", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
