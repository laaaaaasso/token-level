"""Utility helpers for audio significance pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "audio_path",
    "text",
    "gender_label",
    "age_label",
    "emotion_label",
]


DEFAULT_AGE_MAP = {
    "child": "young",
    "youth": "young",
    "teen": "young",
    "young": "young",
    "adult": "old",
    "senior": "old",
    "elder": "old",
    "old": "old",
}


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_outdir(outdir: str | Path) -> Path:
    """Create output directory if missing and return it as Path."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def normalize_label(value: Any) -> Optional[str]:
    """Normalize text-like labels to lower-case strings."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text else None


def load_age_mapping(age_map_json: Optional[str]) -> Dict[str, str]:
    """Load age mapping from JSON file or use defaults."""
    age_map = dict(DEFAULT_AGE_MAP)
    if not age_map_json:
        return age_map
    with open(age_map_json, "r", encoding="utf-8") as f:
        custom = json.load(f)
    for k, v in custom.items():
        nk = normalize_label(k)
        nv = normalize_label(v)
        if nk and nv:
            age_map[nk] = nv
    return age_map


def load_metadata(metadata_path: str | Path, age_map: Mapping[str, str]) -> pd.DataFrame:
    """Load metadata CSV and add normalized label columns."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")

    out = df.copy()
    out["gender_label_norm"] = out["gender_label"].map(normalize_label)
    out["age_label_norm"] = out["age_label"].map(normalize_label)
    out["emotion_label_norm"] = out["emotion_label"].map(normalize_label)
    out["age_label_bin"] = out["age_label_norm"].map(age_map)

    # Keep unresolved labels as their normalized values for flexibility.
    unresolved_mask = out["age_label_bin"].isna()
    out.loc[unresolved_mask, "age_label_bin"] = out.loc[unresolved_mask, "age_label_norm"]

    out["metadata_dir"] = str(path.parent.resolve())
    return out


def resolve_audio_path(audio_path: Any, metadata_dir: str | Path) -> Optional[Path]:
    """Resolve audio path from absolute path or metadata-relative path."""
    if audio_path is None or (isinstance(audio_path, float) and np.isnan(audio_path)):
        return None
    raw = Path(str(audio_path))
    if raw.is_absolute():
        return raw
    return Path(metadata_dir) / raw


def safe_numeric(values: Iterable[Any]) -> np.ndarray:
    """Convert iterable to finite float array."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save JSON with UTF-8 and stable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
