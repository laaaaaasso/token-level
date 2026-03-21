"""Audio feature extraction for significance validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import pandas as pd

from utils import resolve_audio_path

LOGGER = logging.getLogger(__name__)

try:
    import parselmouth  # type: ignore
    from parselmouth.praat import call as praat_call  # type: ignore

    HAS_PARSELMOUTH = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PARSELMOUTH = False


FEATURE_COLUMNS = [
    "duration_sec",
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "f0_range",
    "energy_mean",
    "energy_std",
    "energy_range",
    "tempo_bpm",
    "pause_ratio",
    "spec_centroid_mean",
    "spec_bandwidth_mean",
    "spec_rolloff95_mean",
    "spec_flux_mean",
    "spec_tilt_proxy",
    "mfcc_mean",
    "mfcc_std",
    "mfcc_delta_std",
    "hnr_mean",
    "jitter_local",
    "shimmer_local",
]


def _safe_stat(arr: np.ndarray, op: str) -> float:
    """Return aggregate value or NaN when data is empty."""
    if arr is None or arr.size == 0:
        return np.nan
    if op == "mean":
        return float(np.nanmean(arr))
    if op == "std":
        return float(np.nanstd(arr))
    if op == "min":
        return float(np.nanmin(arr))
    if op == "max":
        return float(np.nanmax(arr))
    raise ValueError(f"unknown op: {op}")


def _extract_optional_parselmouth(audio_path: str) -> Dict[str, float]:
    """Extract optional voice quality features using parselmouth."""
    out = {"hnr_mean": np.nan, "jitter_local": np.nan, "shimmer_local": np.nan}
    if not HAS_PARSELMOUTH:
        return out

    try:
        snd = parselmouth.Sound(audio_path)
        harmonicity = snd.to_harmonicity_cc(time_step=0.01)
        hnr_values = harmonicity.values
        finite_hnr = hnr_values[np.isfinite(hnr_values)]
        out["hnr_mean"] = float(np.mean(finite_hnr)) if finite_hnr.size else np.nan

        point_process = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
        out["jitter_local"] = float(
            praat_call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        )
        out["shimmer_local"] = float(
            praat_call(
                [snd, point_process],
                "Get shimmer (local)",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
        )
    except Exception as exc:  # pragma: no cover - optional best effort
        LOGGER.debug("Parselmouth extraction failed for %s: %s", audio_path, exc)
    return out


def extract_single_audio_features(
    audio_path: str | Path,
    sample_rate: int = 16000,
    top_db: float = 30.0,
) -> Dict[str, float]:
    """Extract all required acoustic features from one audio file."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("empty waveform")

    duration_sec = float(len(y) / sr)

    f0, _, _ = librosa.pyin(
        y,
        fmin=50.0,
        fmax=600.0,
        sr=sr,
    )
    f0_valid = f0[np.isfinite(f0)] if f0 is not None else np.array([])

    f0_mean = _safe_stat(f0_valid, "mean")
    f0_std = _safe_stat(f0_valid, "std")
    f0_min = _safe_stat(f0_valid, "min")
    f0_max = _safe_stat(f0_valid, "max")
    f0_range = f0_max - f0_min if np.isfinite(f0_min) and np.isfinite(f0_max) else np.nan

    rms = librosa.feature.rms(y=y)[0]
    energy_mean = _safe_stat(rms, "mean")
    energy_std = _safe_stat(rms, "std")
    energy_min = _safe_stat(rms, "min")
    energy_max = _safe_stat(rms, "max")
    energy_range = energy_max - energy_min if np.isfinite(energy_min) and np.isfinite(energy_max) else np.nan

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if hasattr(librosa.feature, "tempo"):
        tempo_arr = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    else:
        tempo_arr = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    tempo_bpm = float(tempo_arr[0]) if tempo_arr.size else np.nan

    try:
        nonsilent_intervals = librosa.effects.split(y, top_db=top_db)
        nonsilent = int(np.sum(nonsilent_intervals[:, 1] - nonsilent_intervals[:, 0])) if nonsilent_intervals.size else 0
        pause_ratio = float(max(0.0, 1.0 - nonsilent / max(1, len(y))))
    except Exception:
        if rms.size and np.isfinite(np.max(rms)):
            silence_thresh = 0.1 * float(np.max(rms))
            pause_ratio = float(np.mean(rms <= silence_thresh))
        else:
            pause_ratio = np.nan

    if not np.isfinite(pause_ratio):
        pause_ratio = np.nan

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]

    spec_centroid_mean = _safe_stat(centroid, "mean")
    spec_bandwidth_mean = _safe_stat(bandwidth, "mean")
    spec_rolloff95_mean = _safe_stat(rolloff, "mean")
    spec_flux_mean = _safe_stat(onset_env, "mean")

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, power=2.0)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    low_energy = float(np.mean(mel_db[:13, :])) if mel_db.size else np.nan
    high_energy = float(np.mean(mel_db[-13:, :])) if mel_db.size else np.nan
    spec_tilt_proxy = high_energy - low_energy if np.isfinite(low_energy) and np.isfinite(high_energy) else np.nan

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_mean = _safe_stat(mfcc, "mean")
    mfcc_std = _safe_stat(mfcc, "std")
    mfcc_delta_std = _safe_stat(mfcc_delta, "std")

    out = {
        "duration_sec": duration_sec,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_range": f0_range,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_range": energy_range,
        "tempo_bpm": tempo_bpm,
        "pause_ratio": pause_ratio,
        "spec_centroid_mean": spec_centroid_mean,
        "spec_bandwidth_mean": spec_bandwidth_mean,
        "spec_rolloff95_mean": spec_rolloff95_mean,
        "spec_flux_mean": spec_flux_mean,
        "spec_tilt_proxy": spec_tilt_proxy,
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "mfcc_delta_std": mfcc_delta_std,
        "hnr_mean": np.nan,
        "jitter_local": np.nan,
        "shimmer_local": np.nan,
    }
    out.update(_extract_optional_parselmouth(str(audio_path)))
    return out


def extract_features(
    metadata_df: pd.DataFrame,
    sample_rate: int = 16000,
    top_db: float = 30.0,
) -> pd.DataFrame:
    """Extract audio features for each metadata row."""
    rows: List[Dict[str, Any]] = []
    for idx, row in metadata_df.iterrows():
        resolved_path = resolve_audio_path(row.get("audio_path"), row.get("metadata_dir", "."))
        base: Dict[str, Any] = {"row_id": idx, "resolved_audio_path": str(resolved_path) if resolved_path else None}

        if resolved_path is None or not resolved_path.exists():
            LOGGER.warning("Audio missing, skip extraction: row=%s path=%s", idx, resolved_path)
            empty = {k: np.nan for k in FEATURE_COLUMNS}
            empty["feature_status"] = "missing"
            rows.append({**base, **empty})
            continue

        try:
            features = extract_single_audio_features(
                resolved_path,
                sample_rate=sample_rate,
                top_db=top_db,
            )
            features["feature_status"] = "ok"
            rows.append({**base, **features})
        except Exception as exc:
            LOGGER.warning("Feature extraction failed: row=%s path=%s err=%s", idx, resolved_path, exc)
            empty = {k: np.nan for k in FEATURE_COLUMNS}
            empty["feature_status"] = "error"
            rows.append({**base, **empty})

    feat_df = pd.DataFrame(rows).set_index("row_id")
    feat_df = feat_df.reindex(metadata_df.index)
    return feat_df
