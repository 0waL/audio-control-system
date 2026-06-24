"""
Real-time audio classifier.

Primary backend is a custom PyTorch CNN (`src.cnn_model.AudioCNN`) trained on
log-mel spectrograms. When no trained model is available — or PyTorch isn't
installed — the system falls back to a lightweight MFCC + heuristic classifier
so the real-time pipeline always runs (just with lower accuracy).

PyTorch ships wheels for Python 3.14, so the deep-learning path runs on the
user's existing interpreter (TensorFlow has no 3.14 wheels yet, which is why the
project migrated off YAMNet/Keras).
"""

import numpy as np
import logging
from pathlib import Path
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

CATEGORIES = ["alarm", "speech", "music", "keyboard", "environment"]
CATEGORY_DISPLAY = {
    "alarm": "경보음 / 사이렌",
    "speech": "말소리 (음성)",
    "music": "음악",
    "keyboard": "키보드 / 타이핑",
    "environment": "환경 소음",
}

DEFAULT_CNN_PATH = Path(__file__).parent.parent / "models" / "cnn_torch.pt"


class TorchCNNClassifier:
    """
    PyTorch CNN classifier over log-mel spectrograms.

    Loads a checkpoint saved by training/train_cnn.py (--mode cnn). The
    checkpoint bundles the model weights plus the category list, n_mels, and the
    log-mel normalisation stats (mean/std) used at train time so inference
    matches training exactly.
    """

    def __init__(self, model_path: Optional[str] = None, smoothing_window: int = 5):
        self._model = None
        self._loaded = False
        self._categories = CATEGORIES
        self._n_mels = 64
        self._mean = 0.0
        self._std = 1.0
        self._score_history: deque = deque(maxlen=smoothing_window)
        if model_path:
            self._load(model_path)

    def _load(self, path: str):
        try:
            import torch
            from src.cnn_model import AudioCNN

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self._categories = ckpt.get("categories", CATEGORIES)
            self._n_mels = int(ckpt.get("n_mels", 64))
            self._mean = float(ckpt.get("mean", 0.0))
            self._std = float(ckpt.get("std", 1.0))

            model = AudioCNN(n_mels=self._n_mels, n_classes=len(self._categories))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._model = model
            self._loaded = True
            logger.info(f"PyTorch CNN loaded from {path} "
                        f"(classes={self._categories})")
        except Exception as e:
            logger.warning(f"PyTorch CNN unavailable ({e}), using fallback classifier")
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[dict]:
        if not self._loaded:
            return None
        try:
            import torch
            from data.preprocessing import extract_mel_spectrogram

            mel = extract_mel_spectrogram(audio, sr=sample_rate, n_mels=self._n_mels)
            mel = (mel - self._mean) / (self._std + 1e-6)
            # (n_mels, frames) → (1, 1, n_mels, frames)
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                logits = self._model(x)
                probs_t = torch.softmax(logits, dim=1)[0].numpy()

            probs = {c: float(p) for c, p in zip(self._categories, probs_t)}
            # Pad any categories the model wasn't trained on, for a stable schema.
            for c in CATEGORIES:
                probs.setdefault(c, 0.0)

            self._score_history.append(probs)
            smoothed = {c: float(np.mean([h[c] for h in self._score_history]))
                        for c in CATEGORIES}
            return smoothed
        except Exception as e:
            logger.error(f"PyTorch CNN inference error: {e}")
            return None


class MFCCClassifier:
    """
    Lightweight MFCC + heuristic classifier used when the CNN is unavailable
    or as a fast pre-filter.

    Features used:
      - MFCC mean/std (13 coefficients)
      - Spectral centroid, rolloff, bandwidth
      - Zero crossing rate
      - RMS energy
    """

    def __init__(self, model_path: Optional[str] = None, smoothing_window: int = 5):
        self._model = None
        self._smoothing_window = smoothing_window
        self._score_history: deque = deque(maxlen=smoothing_window)
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        try:
            import pickle
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            logger.info(f"MFCC classifier loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load MFCC model ({e}), using heuristics")

    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa
        features = []

        # MFCC (13 coefficients, mean + std = 26 values)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        features.append(float(np.mean(centroid)))
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        features.append(float(np.mean(rolloff)))
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        features.append(float(np.mean(bandwidth)))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(float(np.mean(zcr)))

        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features.append(float(np.mean(rms)))

        return np.array(features, dtype=np.float32)

    def _heuristic_classify(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Rule-based classification using acoustic features.
        Used when no trained model is available.
        """
        import librosa

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.001:
            # Silence — treat as environment noise at very low probability
            return {cat: 0.2 for cat in CATEGORIES}

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        ))
        rolloff = float(np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=sample_rate,
                                             roll_percent=0.85)
        ))
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        scores = {cat: 0.0 for cat in CATEGORIES}

        # Alarm: high energy, high frequency content, abrupt onset
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onset_strength = float(np.max(onset_env))
        if centroid > 3000 and rms > 0.05 and onset_strength > 5.0:
            scores["alarm"] += 0.6
        if rolloff > 8000 and rms > 0.04:
            scores["alarm"] += 0.2

        # Speech: moderate ZCR, centroid 100–4000 Hz, MFCC variance typical
        if 0.03 < zcr < 0.25 and 500 < centroid < 4000:
            scores["speech"] += 0.5
        if np.std(mfcc_mean) > 15:
            scores["speech"] += 0.15

        # Music: low ZCR, tonal content, harmonic richness
        harmonic, _ = librosa.effects.hpss(audio)
        harmonic_ratio = float(np.mean(np.abs(harmonic))) / (rms + 1e-6)
        if harmonic_ratio > 0.6 and zcr < 0.1:
            scores["music"] += 0.5
        if 100 < centroid < 3000 and zcr < 0.08:
            scores["music"] += 0.2

        # Keyboard: very transient, mid-frequency bursts, low sustained energy
        if onset_strength > 3.0 and rms < 0.03 and zcr > 0.15:
            scores["keyboard"] += 0.5
        if centroid > 2000 and rms < 0.02:
            scores["keyboard"] += 0.2

        # Environment: diffuse, lower energy, spread spectrum
        if rms < 0.02 and zcr < 0.15:
            scores["environment"] += 0.4
        if centroid < 2000 and onset_strength < 2.0:
            scores["environment"] += 0.2

        # Normalize
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def classify(self, audio: np.ndarray, sample_rate: int = 22050) -> dict:
        if self._model is not None:
            try:
                features = self._extract_features(audio, sample_rate)
                probs = self._model.predict_proba([features])[0]
                result = dict(zip(CATEGORIES, probs))
            except Exception as e:
                logger.warning(f"Model inference failed ({e}), using heuristics")
                result = self._heuristic_classify(audio, sample_rate)
        else:
            result = self._heuristic_classify(audio, sample_rate)

        self._score_history.append(result)
        smoothed = {cat: np.mean([h[cat] for h in self._score_history])
                    for cat in CATEGORIES}
        return smoothed


class AudioClassifier:
    """
    Unified classifier that tries the PyTorch CNN first, then falls back to the
    MFCC classifier (trained sklearn model if provided, else heuristics).
    Provides a single classify() interface to the rest of the system.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        detection_threshold: float = 0.3,
        mfcc_model_path: Optional[str] = None,
        cnn_model_path: Optional[str] = None,
        use_cnn: bool = True,
    ):
        self._threshold = detection_threshold
        self._cnn: Optional[TorchCNNClassifier] = None
        self._mfcc = MFCCClassifier(
            model_path=mfcc_model_path,
            smoothing_window=smoothing_window,
        )

        if use_cnn:
            path = cnn_model_path
            if path is None and DEFAULT_CNN_PATH.exists():
                path = str(DEFAULT_CNN_PATH)
            if path:
                self._cnn = TorchCNNClassifier(
                    model_path=path, smoothing_window=smoothing_window
                )
            else:
                logger.info(
                    "No trained CNN found (models/cnn_torch.pt); using MFCC "
                    "fallback. Train one with: python training/train_cnn.py --mode cnn"
                )

        self._last_result: Optional[dict] = None

    def classify(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Classify audio chunk and return probability dict over CATEGORIES.
        Automatically selects the best available backend.
        """
        result = None

        if self._cnn and self._cnn.is_loaded():
            result = self._cnn.classify(audio, sample_rate)

        if result is None:
            result = self._mfcc.classify(audio, sample_rate)

        self._last_result = result
        return result

    def top_category(self, probs: dict) -> tuple[str, float]:
        """Return (category_name, confidence) for the highest-probability class."""
        best = max(probs, key=probs.get)
        return best, probs[best]

    def is_significant(self, probs: dict) -> bool:
        """Return True if any category exceeds the detection threshold."""
        return any(v >= self._threshold for v in probs.values())

    @property
    def last_result(self) -> Optional[dict]:
        return self._last_result

    @staticmethod
    def category_display_name(category: str) -> str:
        return CATEGORY_DISPLAY.get(category, category)
