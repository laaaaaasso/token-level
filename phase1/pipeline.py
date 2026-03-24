"""Phase 1 pipeline: prepare D_ref and train CosyVoice2 reference LM."""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


@dataclass
class Phase1DataStats:
    total_rows: int
    kept_rows: int
    dropped_rows: int
    train_rows: int
    cv_rows: int


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def _prepare_ref_rows(
    ref_manifest: Path,
    seed: int,
    val_ratio: float,
    max_samples: int | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Phase1DataStats]:
    rows = _read_jsonl(ref_manifest)
    if max_samples is not None:
        rows = rows[: max(0, max_samples)]

    kept: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        mu = row.get("mu", [])
        audio_path = str(row.get("audio_path", "")).strip()
        status = str(row.get("status", "ok")).strip().lower()
        if status not in {"ok", ""}:
            continue
        if not text or not audio_path:
            continue
        if not isinstance(mu, list) or len(mu) == 0:
            continue
        item = dict(row)
        item["sample_id"] = str(item.get("sample_id", f"phase1_ref_{i:07d}"))
        kept.append(item)

    if not kept:
        raise RuntimeError("No valid D_ref rows found in ref manifest.")

    rng = random.Random(seed)
    idx = list(range(len(kept)))
    rng.shuffle(idx)

    if len(kept) == 1:
        train_rows = [kept[0]]
        cv_rows = [kept[0]]
    else:
        raw_cv = int(round(len(kept) * max(0.0, val_ratio)))
        cv_count = min(len(kept) - 1, max(1, raw_cv))
        cv_index = set(idx[:cv_count])
        train_rows = [kept[i] for i in range(len(kept)) if i not in cv_index]
        cv_rows = [kept[i] for i in range(len(kept)) if i in cv_index]

    stats = Phase1DataStats(
        total_rows=len(rows),
        kept_rows=len(kept),
        dropped_rows=len(rows) - len(kept),
        train_rows=len(train_rows),
        cv_rows=len(cv_rows),
    )
    return train_rows, cv_rows, stats


def _iter_parquet_rows(
    rows: Iterable[Dict[str, Any]],
    embedding_dim: int = 192,
) -> Iterable[Dict[str, Any]]:
    zero_embedding = [0.0] * embedding_dim
    for row in rows:
        audio_path = Path(str(row["audio_path"])).resolve()
        if not audio_path.exists():
            continue
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        yield {
            "utt": str(row["sample_id"]),
            "text": str(row["text"]),
            "audio_data": audio_bytes,
            "speech_token": [int(x) for x in row["mu"]],
            # Keep embeddings in parquet to avoid online embedding extraction.
            "utt_embedding": zero_embedding,
            "spk_embedding": zero_embedding,
        }


def _write_parquet_shards(
    rows: Sequence[Dict[str, Any]],
    parquet_dir: Path,
    prefix: str,
    num_utts_per_parquet: int,
) -> List[Path]:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Path] = []
    chunk: List[Dict[str, Any]] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal chunk, shard_idx
        if not chunk:
            return
        shard_path = parquet_dir / f"{prefix}_{shard_idx:05d}.parquet"
        table = pa.Table.from_pylist(chunk)
        pq.write_table(table, shard_path)
        shards.append(shard_path.resolve())
        shard_idx += 1
        chunk = []

    for sample in _iter_parquet_rows(rows):
        chunk.append(sample)
        if len(chunk) >= max(1, num_utts_per_parquet):
            flush()
    flush()
    return shards


def _patch_cosyvoice2_train_conf(
    base_config: Path,
    output_config: Path,
    learning_rate: float,
    max_epoch: int,
    log_interval: int,
    save_per_step: int,
) -> None:
    lines = base_config.read_text(encoding="utf-8").splitlines(keepends=True)
    in_train_conf = False
    in_optim_conf = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_top_level = bool(line) and (not line.startswith(" ") and not line.startswith("\t"))

        if stripped == "train_conf:":
            in_train_conf = True
            in_optim_conf = False
            continue

        if in_train_conf and is_top_level:
            in_train_conf = False
            in_optim_conf = False

        if not in_train_conf:
            continue

        if re.match(r"^\s{4}optim_conf:\s*$", line):
            in_optim_conf = True
            continue
        if re.match(r"^\s{4}[A-Za-z_]\w*:\s*", line) and not re.match(r"^\s{4}optim_conf:\s*$", line):
            in_optim_conf = False

        if in_optim_conf and re.match(r"^\s{8}lr:\s*", line):
            lines[i] = f"        lr: {learning_rate}\n"
            continue
        if re.match(r"^\s{4}max_epoch:\s*", line):
            lines[i] = f"    max_epoch: {int(max_epoch)}\n"
            continue
        if re.match(r"^\s{4}log_interval:\s*", line):
            lines[i] = f"    log_interval: {int(log_interval)}\n"
            continue
        if re.match(r"^\s{4}save_per_step:\s*", line):
            lines[i] = f"    save_per_step: {int(save_per_step)}\n"
            continue

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text("".join(lines), encoding="utf-8")


def _find_best_and_last_checkpoints(model_dir: Path) -> Tuple[Path | None, Path | None]:
    all_ckpts = sorted(model_dir.glob("epoch_*_whole.pt"))
    if not all_ckpts:
        return None, None

    last_ckpt = all_ckpts[-1]
    best_ckpt: Path | None = None
    best_loss: float | None = None

    for ckpt in all_ckpts:
        sidecar = ckpt.with_suffix(".yaml")
        if not sidecar.exists():
            continue
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            loss_dict = meta.get("loss_dict", {})
            loss_value = loss_dict.get("loss", None)
            if loss_value is None:
                continue
            loss_value = float(loss_value)
        except Exception:
            continue
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_ckpt = ckpt

    if best_ckpt is None:
        best_ckpt = last_ckpt
    return best_ckpt, last_ckpt


def _export_frozen_rm(model_dir: Path, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt, last_ckpt = _find_best_and_last_checkpoints(model_dir)
    if best_ckpt is None or last_ckpt is None:
        return {}

    rm_best = output_dir / "rm_best.pt"
    rm_last = output_dir / "rm_last.pt"
    rm_frozen = output_dir / "rm_frozen.pt"
    shutil.copy2(best_ckpt, rm_best)
    shutil.copy2(last_ckpt, rm_last)
    shutil.copy2(best_ckpt, rm_frozen)
    return {
        "rm_best": str(rm_best.resolve()),
        "rm_last": str(rm_last.resolve()),
        "rm_frozen": str(rm_frozen.resolve()),
        "best_source": str(best_ckpt.resolve()),
        "last_source": str(last_ckpt.resolve()),
    }


def run_phase1_training(
    ref_manifest: str,
    output_dir: str,
    cosyvoice_root: str = "CosyVoice",
    base_config: str = "CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml",
    qwen_pretrain_path: str = "",
    init_checkpoint: str | None = None,
    onnx_path: str = ".",
    learning_rate: float = 1e-5,
    max_epoch: int = 3,
    val_ratio: float = 0.05,
    log_interval: int = 20,
    save_per_step: int = -1,
    seed: int = 42,
    max_samples: int | None = None,
    num_utts_per_parquet: int = 500,
    num_gpus: int = 1,
    num_workers: int = 0,
    prefetch: int = 16,
    pin_memory: bool = False,
    use_amp: bool = False,
    run_training: bool = False,
) -> Dict[str, Any]:
    outdir = Path(output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ref_manifest_path = Path(ref_manifest).resolve()
    if not ref_manifest_path.exists():
        raise FileNotFoundError(f"ref manifest not found: {ref_manifest_path}")

    train_rows, cv_rows, stats = _prepare_ref_rows(
        ref_manifest=ref_manifest_path,
        seed=seed,
        val_ratio=val_ratio,
        max_samples=max_samples,
    )

    parquet_root = outdir / "phase1_ref_parquet"
    train_parquet = _write_parquet_shards(
        train_rows, parquet_root / "train", prefix="train", num_utts_per_parquet=num_utts_per_parquet
    )
    cv_parquet = _write_parquet_shards(
        cv_rows, parquet_root / "cv", prefix="cv", num_utts_per_parquet=num_utts_per_parquet
    )
    if not train_parquet or not cv_parquet:
        raise RuntimeError("Failed to generate parquet shards from D_ref.")

    train_list = outdir / "phase1_ref_train.data.list"
    cv_list = outdir / "phase1_ref_cv.data.list"
    _write_lines(train_list, [str(p) for p in train_parquet])
    _write_lines(cv_list, [str(p) for p in cv_parquet])

    base_config_path = Path(base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"base config not found: {base_config_path}")
    phase1_config = outdir / "phase1_cosyvoice2_ref.yaml"
    _patch_cosyvoice2_train_conf(
        base_config=base_config_path,
        output_config=phase1_config,
        learning_rate=learning_rate,
        max_epoch=max_epoch,
        log_interval=log_interval,
        save_per_step=save_per_step,
    )

    model_dir = outdir / "checkpoints"
    tensorboard_dir = outdir / "tensorboard"
    cosyvoice_root_path = Path(cosyvoice_root).resolve()
    train_py = cosyvoice_root_path / "cosyvoice" / "bin" / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"CosyVoice train.py not found: {train_py}")

    cmd: List[str] = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={int(max(1, num_gpus))}",
        str(train_py),
        "--train_engine",
        "torch_ddp",
        "--model",
        "llm",
        "--config",
        str(phase1_config),
        "--train_data",
        str(train_list),
        "--cv_data",
        str(cv_list),
        "--onnx_path",
        str(onnx_path),
        "--model_dir",
        str(model_dir),
        "--tensorboard_dir",
        str(tensorboard_dir),
        "--ddp.dist_backend",
        "nccl",
        "--num_workers",
        str(int(max(0, num_workers))),
        "--prefetch",
        str(int(max(1, prefetch))),
    ]
    if str(qwen_pretrain_path).strip():
        cmd.extend(["--qwen_pretrain_path", str(qwen_pretrain_path)])
    if pin_memory:
        cmd.append("--pin_memory")
    if use_amp:
        cmd.append("--use_amp")
    if init_checkpoint:
        cmd.extend(["--checkpoint", str(init_checkpoint)])

    cmd_text = " ".join(cmd)
    cmd_file = outdir / "phase1_train_command.txt"
    _write_lines(cmd_file, [cmd_text])

    train_returncode: int | None = None
    train_skipped_reason: str | None = None
    if run_training:
        if not str(qwen_pretrain_path).strip():
            raise ValueError("run_training=True requires --qwen-pretrain-path.")
        process = subprocess.run(cmd, cwd=str(cosyvoice_root_path), check=False)
        train_returncode = int(process.returncode)
    else:
        train_skipped_reason = "run_training is False (prepared data/config/command only)."

    export_info: Dict[str, str] = {}
    if train_returncode == 0:
        export_info = _export_frozen_rm(model_dir=model_dir, output_dir=outdir)

    summary: Dict[str, Any] = {
        "phase": "phase1_reference_lm",
        "input_ref_manifest": str(ref_manifest_path),
        "data_stats": {
            "total_rows": stats.total_rows,
            "kept_rows": stats.kept_rows,
            "dropped_rows": stats.dropped_rows,
            "train_rows": stats.train_rows,
            "cv_rows": stats.cv_rows,
        },
        "artifacts": {
            "train_data_list": str(train_list),
            "cv_data_list": str(cv_list),
            "phase1_config": str(phase1_config),
            "train_command_file": str(cmd_file),
            "model_dir": str(model_dir),
            "tensorboard_dir": str(tensorboard_dir),
        },
        "training": {
            "run_training": run_training,
            "train_returncode": train_returncode,
            "train_skipped_reason": train_skipped_reason,
            "command": cmd_text,
        },
        "rm_export": export_info,
    }
    _write_json(outdir / "phase1_train_summary.json", summary)
    return summary
