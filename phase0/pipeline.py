"""Phase 0 pipeline: data preparation and dataset layering."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .tokenizers import ByteTextTokenizer, CosyVoice2SpeechTokenizer

LOGGER = logging.getLogger(__name__)


def _load_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ValueError(f"unsupported manifest format: {p.suffix}")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_text(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    text = str(v).strip().lower()
    return text in {"1", "true", "yes", "y", "ref", "high", "high_score"}


def _resolve_audio_path(
    raw_audio_path: str,
    manifest_dir: Path,
    audio_base_dir: Optional[Path],
) -> Path:
    raw = Path(raw_audio_path)
    if raw.is_absolute():
        return raw
    if audio_base_dir is not None:
        return (audio_base_dir / raw).resolve()
    return (manifest_dir / raw).resolve()


def _prepare_train_df(
    train_df: pd.DataFrame,
    train_manifest_path: Path,
    audio_base_dir: Optional[Path],
) -> pd.DataFrame:
    required_cols = ["audio_path", "text"]
    missing = [c for c in required_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"train manifest missing columns: {missing}")

    out = train_df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = [f"sample_{i:07d}" for i in range(len(out))]
    else:
        out["sample_id"] = out["sample_id"].map(_to_text)
        missing_id_mask = out["sample_id"] == ""
        if missing_id_mask.any():
            fill_ids = [f"sample_{i:07d}" for i in out[missing_id_mask].index]
            out.loc[missing_id_mask, "sample_id"] = fill_ids

    out["audio_path_raw"] = out["audio_path"].map(_to_text)
    manifest_dir = train_manifest_path.parent.resolve()
    out["audio_path_abs"] = out["audio_path_raw"].map(
        lambda x: str(_resolve_audio_path(x, manifest_dir, audio_base_dir))
    )
    out["text"] = out["text"].map(_to_text)
    return out


def _build_ref_membership(
    train_df: pd.DataFrame,
    train_manifest_path: Path,
    ref_manifest_path: Optional[str | Path],
    ref_flag_column: Optional[str],
    audio_base_dir: Optional[Path],
) -> Dict[str, Any]:
    ref_flags = pd.Series(False, index=train_df.index)
    meta: Dict[str, Any] = {
        "ref_source": [],
        "ref_match_mode": None,
        "ref_rows": 0,
        "ref_unmatched_count": 0,
    }

    if ref_manifest_path:
        ref_df = _load_manifest(ref_manifest_path).copy()
        meta["ref_rows"] = int(len(ref_df))
        meta["ref_source"].append("ref_manifest")
        if ("sample_id" not in ref_df.columns) and ("audio_path" not in ref_df.columns):
            raise ValueError("ref manifest must contain sample_id or audio_path")

        if "sample_id" in ref_df.columns:
            ref_df["sample_id"] = ref_df["sample_id"].map(_to_text)
        if "audio_path" in ref_df.columns:
            ref_df["audio_path"] = ref_df["audio_path"].map(_to_text)
        if "resolved_audio_path" in ref_df.columns:
            ref_df["resolved_audio_path"] = ref_df["resolved_audio_path"].map(_to_text)
        if "metadata_dir" in ref_df.columns:
            ref_df["metadata_dir"] = ref_df["metadata_dir"].map(_to_text)

        if "sample_id" in ref_df.columns and train_df["sample_id"].notna().any():
            ref_ids = set(ref_df["sample_id"][ref_df["sample_id"] != ""].tolist())
            ref_flags = ref_flags | train_df["sample_id"].isin(ref_ids)
            meta["ref_match_mode"] = "sample_id"
            meta["ref_unmatched_count"] = int(len(ref_ids) - int(train_df["sample_id"].isin(ref_ids).sum()))
        else:
            # Prefer resolved paths when available in ref manifest.
            if "resolved_audio_path" in ref_df.columns and (ref_df["resolved_audio_path"] != "").any():
                ref_abs = ref_df["resolved_audio_path"]
            elif "metadata_dir" in ref_df.columns and "audio_path" in ref_df.columns:
                ref_abs = ref_df.apply(
                    lambda r: str((Path(r["metadata_dir"]) / r["audio_path"]).resolve()),
                    axis=1,
                )
            else:
                ref_manifest_dir = Path(ref_manifest_path).parent.resolve()
                ref_abs = ref_df["audio_path"].map(
                    lambda x: str(_resolve_audio_path(x, ref_manifest_dir, audio_base_dir))
                )

            ref_paths = set(ref_abs.tolist())
            ref_flags = ref_flags | train_df["audio_path_abs"].isin(ref_paths)
            meta["ref_match_mode"] = "audio_path"
            meta["ref_unmatched_count"] = int(len(ref_paths) - int(train_df["audio_path_abs"].isin(ref_paths).sum()))

    if ref_flag_column:
        if ref_flag_column not in train_df.columns:
            raise ValueError(f"ref flag column not found in train manifest: {ref_flag_column}")
        flag_series = train_df[ref_flag_column].map(_to_bool)
        ref_flags = ref_flags | flag_series
        meta["ref_source"].append(f"train_column:{ref_flag_column}")

    if not ref_manifest_path and not ref_flag_column:
        raise ValueError("one of --ref-manifest or --ref-flag-column is required")

    return {"flags": ref_flags, "meta": meta}


def run_phase0_pipeline(
    train_manifest: str | Path,
    outdir: str | Path,
    ref_manifest: Optional[str | Path] = None,
    ref_flag_column: Optional[str] = None,
    audio_base_dir: Optional[str | Path] = None,
    speech_sample_rate: int = 16000,
    speech_vocab_size: int = 1024,
) -> Dict[str, Any]:
    """Run complete Phase 0 data preparation pipeline."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    audio_base = Path(audio_base_dir).resolve() if audio_base_dir else None

    train_manifest_path = Path(train_manifest).resolve()
    train_df = _prepare_train_df(
        _load_manifest(train_manifest_path),
        train_manifest_path=train_manifest_path,
        audio_base_dir=audio_base,
    )

    ref_info = _build_ref_membership(
        train_df=train_df,
        train_manifest_path=train_manifest_path,
        ref_manifest_path=ref_manifest,
        ref_flag_column=ref_flag_column,
        audio_base_dir=audio_base,
    )
    train_df["in_ref"] = ref_info["flags"]
    train_df["split"] = train_df["in_ref"].map(lambda x: "ref" if bool(x) else "train")

    text_tokenizer = ByteTextTokenizer()
    speech_tokenizer = CosyVoice2SpeechTokenizer(
        sample_rate=speech_sample_rate,
        vocab_size=speech_vocab_size,
    )

    records_all: List[Dict[str, Any]] = []
    records_ok: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()

    total = len(train_df)
    LOGGER.info("Phase0 start: total train samples=%s", total)

    for i, row in train_df.iterrows():
        sample_id = _to_text(row["sample_id"])
        text = _to_text(row["text"])
        audio_path_abs = Path(_to_text(row["audio_path_abs"]))
        in_ref = bool(row["in_ref"])
        split = _to_text(row["split"])

        status = "ok"
        error = ""
        y: List[int] = []
        mu: List[int] = []
        duration_sec = 0.0

        if not audio_path_abs.exists():
            status = "failed"
            error = "audio_not_found"
        elif text == "":
            status = "failed"
            error = "text_empty"
        else:
            y = text_tokenizer.encode(text)
            if len(y) == 0:
                status = "failed"
                error = "text_tokens_empty"
            else:
                try:
                    mu, duration_sec = speech_tokenizer.encode(audio_path_abs)
                except Exception as exc:
                    status = "failed"
                    error = f"speech_tokenizer_error:{type(exc).__name__}"
                if status == "ok" and len(mu) == 0:
                    status = "failed"
                    error = "speech_tokens_empty"

        item = {
            "sample_id": sample_id,
            "audio_path": str(audio_path_abs),
            "text": text,
            "split": split,
            "in_ref": in_ref,
            "y": y,
            "mu": mu,
            "duration_sec": duration_sec,
            "status": status,
            "error": error,
        }
        records_all.append(item)

        if status == "ok":
            records_ok.append(item)
        else:
            failure_reasons[error] += 1
            failures.append(item)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            LOGGER.info("Phase0 progress: %s/%s", i + 1, total)

    train_ok = [r for r in records_ok if r["split"] in {"train", "ref"}]
    ref_ok = [r for r in records_ok if r["in_ref"]]

    _write_jsonl(out_path / "phase0_manifest_all.jsonl", records_all)
    _write_jsonl(out_path / "phase0_manifest_train.jsonl", train_ok)
    _write_jsonl(out_path / "phase0_manifest_ref.jsonl", ref_ok)
    _write_jsonl(out_path / "phase0_failures.jsonl", failures)

    avg_y = float(sum(len(r["y"]) for r in records_ok) / len(records_ok)) if records_ok else 0.0
    avg_mu = float(sum(len(r["mu"]) for r in records_ok) / len(records_ok)) if records_ok else 0.0

    summary = {
        "total_input_samples": int(total),
        "total_train_marked": int((train_df["split"] == "train").sum()),
        "total_ref_marked": int((train_df["split"] == "ref").sum()),
        "successful_samples": int(len(records_ok)),
        "failed_samples": int(len(failures)),
        "avg_text_tokens_per_success": avg_y,
        "avg_speech_tokens_per_success": avg_mu,
        "failure_reasons": dict(failure_reasons),
        "ref_meta": ref_info["meta"],
        "tokenizer": {
            "text_tokenizer": "byte_utf8",
            "speech_tokenizer": "CosyVoice2SpeechTokenizer(default_mock_backend)",
            "speech_sample_rate": speech_sample_rate,
            "speech_vocab_size": speech_vocab_size,
        },
        "outputs": {
            "manifest_all": str((out_path / "phase0_manifest_all.jsonl").resolve()),
            "manifest_train": str((out_path / "phase0_manifest_train.jsonl").resolve()),
            "manifest_ref": str((out_path / "phase0_manifest_ref.jsonl").resolve()),
            "failures": str((out_path / "phase0_failures.jsonl").resolve()),
        },
    }

    with open(out_path / "phase0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Phase0 done: success=%s, failed=%s, ref_marked=%s",
        len(records_ok),
        len(failures),
        int((train_df["split"] == "ref").sum()),
    )
    return summary
