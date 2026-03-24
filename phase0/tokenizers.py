"""Tokenizers used in Phase 0 data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np


class ByteTextTokenizer:
    """Simple text tokenizer that maps text to UTF-8 byte ids."""

    def encode(self, text: str) -> List[int]:
        if text is None:
            return []
        return list(str(text).encode("utf-8"))


@dataclass
class MockCosyVoice2SpeechTokenizer:
    """
    Default speech tokenizer used in Phase 0.

    This class provides a stable speech-token interface and can be replaced by
    a real CosyVoice2 tokenizer later without changing pipeline logic.
    """

    sample_rate: int = 16000
    hop_length: int = 320
    n_fft: int = 1024
    n_mels: int = 64
    vocab_size: int = 1024

    def encode(self, audio_path: str | Path) -> Tuple[List[int], float]:
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        if y.size == 0:
            return [], 0.0

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        log_mel = np.log1p(mel)
        frame_feature = log_mel.mean(axis=0)

        if frame_feature.size == 0:
            return [], float(len(y) / sr)

        low = float(np.percentile(frame_feature, 1))
        high = float(np.percentile(frame_feature, 99))
        if high <= low:
            tokens = np.zeros(frame_feature.shape[0], dtype=np.int64)
        else:
            clipped = np.clip(frame_feature, low, high)
            norm = (clipped - low) / (high - low + 1e-8)
            tokens = np.round(norm * (self.vocab_size - 1)).astype(np.int64)

        return tokens.tolist(), float(len(y) / sr)


# Keep this alias for clearer future replacement.
CosyVoice2SpeechTokenizer = MockCosyVoice2SpeechTokenizer
