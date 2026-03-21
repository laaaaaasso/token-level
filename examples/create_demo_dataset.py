"""Create a small synthetic dataset for quick pipeline smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprised"]


def synth_voice(
    sr: int,
    duration: float,
    base_f0: float,
    energy_scale: float,
    modulation: float,
    pause_ratio: float,
) -> np.ndarray:
    """Generate a simple voice-like signal."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    f0_track = base_f0 * (1.0 + modulation * np.sin(2 * np.pi * 2.0 * t))
    phase = 2 * np.pi * np.cumsum(f0_track) / sr
    signal = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.random.randn(len(t))
    signal = energy_scale * signal

    frame = int(sr * 0.05)
    if frame > 0:
        n_frames = len(signal) // frame
        n_sil = int(n_frames * pause_ratio)
        if n_sil > 0:
            silent_idx = np.random.choice(n_frames, size=n_sil, replace=False)
            for idx in silent_idx:
                start = idx * frame
                signal[start : start + frame] *= 0.05

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.95 * signal / peak
    return signal.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Directory to write demo dataset.")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    audio_dir = outdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    i = 0
    for emo in EMOTIONS:
        for rep in range(4):
            gender = "male" if (rep % 2 == 0) else "female"
            age = "young" if (rep < 2) else "old"
            base_f0 = 120 if gender == "male" else 220
            if age == "young":
                base_f0 += 20
            else:
                base_f0 -= 5

            if emo == "happy":
                energy_scale, modulation = 1.0, 0.15
            elif emo == "sad":
                energy_scale, modulation = 0.55, 0.05
            elif emo == "angry":
                energy_scale, modulation = 1.2, 0.2
            elif emo == "fear":
                energy_scale, modulation = 0.9, 0.25
            elif emo == "surprised":
                energy_scale, modulation = 1.1, 0.18
            else:
                energy_scale, modulation = 0.8, 0.08

            pause_ratio = 0.08 if age == "young" else 0.2
            audio = synth_voice(
                sr=args.sr,
                duration=2.0 + 0.2 * rep,
                base_f0=base_f0,
                energy_scale=energy_scale,
                modulation=modulation,
                pause_ratio=pause_ratio,
            )

            audio_name = f"demo_{i:03d}_{emo}_{gender}_{age}.wav"
            audio_path = audio_dir / audio_name
            sf.write(audio_path, audio, args.sr)
            rows.append(
                {
                    "audio_path": str(Path("audio") / audio_name),
                    "text": f"synthetic sample {i}",
                    "gender_label": gender,
                    "age_label": age,
                    "emotion_label": emo,
                }
            )
            i += 1

    metadata = pd.DataFrame(rows)
    metadata.to_csv(outdir / "metadata.csv", index=False, encoding="utf-8-sig")
    print(f"Demo dataset created: {outdir}")


if __name__ == "__main__":
    main()
