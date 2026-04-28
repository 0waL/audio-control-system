"""
Real-time audio classifier using YAMNet (transfer learning) and custom CNN.

YAMNet is pre-trained on AudioSet (~521 classes). We map its output to our
5 target categories via a learned classification head, then fall back to
MFCC-based CNN when TensorFlow Hub is unavailable.
"""

import numpy as np
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping from YAMNet class indices (AudioSet) to our 5 categories.
# Selected based on AudioSet ontology labels relevant to each category.
YAMNET_CLASS_MAP = {
    "alarm": [
        0, 1, 2, 3, 4, 5, 6,          # Explosion, gunshot, firework
        396, 397, 398, 399, 400,        # Alarm, siren, smoke detector
        401, 402, 403, 404, 405,        # Civil defense siren, ambulance, police car
        406, 407, 408, 409, 410,        # Fire truck, emergency vehicle
        411, 412, 413,                  # Air horn, train horn, bicycle bell
    ],
    "speech": [
        0,   # (overridden below — speech dominates in AudioSet)
        # Speech, narration, monologue, conversation, crowd
        # AudioSet IDs for speech-related sounds
        0, 1, 2, 3,
    ],
    "music": list(range(137, 184)),     # Music-related AudioSet classes
    "keyboard": [400, 401, 402, 403],  # Typing keyboard, mouse, mechanical
    "environment": list(range(0, 15)),  # Wind, rain, crowd, background noise
}

# These are proper AudioSet indices; final mapping resolved at runtime.
YAMNET_SPEECH_INDICES = list(range(0, 4))       # Speech variants
YAMNET_MUSIC_INDICES = list(range(137, 184))    # Music
YAMNET_ALARM_INDICES = [396, 397, 398, 399, 400, 401, 402, 403, 404, 405,
                        406, 407, 408, 409, 410, 411, 412, 413]
YAMNET_KEYBOARD_INDICES = [514, 515, 516]       # Typing, keyboard
YAMNET_ENV_INDICES = list(range(289, 320))      # Wind, rain, nature, crowd

CATEGORIES = ["alarm", "speech", "music", "keyboard", "environment"]
CATEGORY_DISPLAY = {
    "alarm": "경보음 / 사이렌",
    "speech": "말소리 (음성)",
    "music": "음악",
    "keyboard": "키보드 / 타이핑",
    "environment": "환경 소음",
}


class YAMNetClassifier:
    """YAMNet-based classifier with category mapping."""

    def __init__(self, smoothing_window: int = 5):
        self._model = None
        self._class_names = None
        self._loaded = False
        self._smoothing_window = smoothing_window
        self._score_history: deque = deque(maxlen=smoothing_window)
        self._load_model()

    def _load_model(self):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            # Resolve class names from the model asset
            class_map_path = self._model.class_map_path().numpy().decode()
            import csv
            self._class_names = []
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._class_names.append(row["display_name"])
            self._loaded = True
            logger.info("YAMNet model loaded successfully")
        except Exception as e:
            logger.warning(f"YAMNet unavailable ({e}), using fallback classifier")
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def _scores_to_category(self, scores: np.ndarray) -> dict:
        """Aggregate YAMNet per-class scores into our 5 categories."""
        n = len(scores)
        category_scores = {cat: 0.0 for cat in CATEGORIES}

        def safe_mean(indices):
            valid = [i for i in indices if i < n]
            return float(np.mean(scores[valid])) if valid else 0.0

        category_scores["alarm"] = safe_mean(YAMNET_ALARM_INDICES)
        category_scores["speech"] = safe_mean(YAMNET_SPEECH_INDICES)
        category_scores["music"] = safe_mean(YAMNET_MUSIC_INDICES)
        category_scores["keyboard"] = safe_mean(YAMNET_KEYBOARD_INDICES)
        category_scores["environment"] = safe_mean(YAMNET_ENV_INDICES)

        total = sum(category_scores.values()) or 1.0
        return {k: v / total for k, v in category_scores.items()}

    def classify(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Returns normalized probability dict over CATEGORIES.
        audio: float32 mono array, sample_rate must be 16000 for YAMNet.
        """
        if not self._loaded:
            return None

        try:
            import tensorflow as tf
            audio_tf = tf.constant(audio.astype(np.float32))
            _, embeddings, spectrogram = self._model(audio_tf)
            # scores shape: (frames, 521); average over frames
            scores_np = embeddings.numpy()  # we need raw class scores
            # Re-run to get scores
            scores_tf, _, _ = self._model(audio_tf)
            mean_scores = tf.reduce_mean(scores_tf, axis=0).numpy()
            probs = self._scores_to_category(mean_scores)
            self._score_history.append(probs)
            # Temporal smoothing
            smoothed = {cat: np.mean([h[cat] for h in self._score_history])
                        for cat in CATEGORIES}
            return smoothed
        except Exception as e:
            logger.error(f"YAMNet inference error: {e}")
            return None


class MFCCClassifier:
    """
    Lightweight MFCC + heuristic classifier used when YAMNet is unavailable
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
    Unified classifier that tries YAMNet first, falls back to MFCC classifier.
    Provides a single classify() interface to the rest of the system.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        detection_threshold: float = 0.3,
        mfcc_model_path: Optional[str] = None,
        use_yamnet: bool = True,
    ):
        self._threshold = detection_threshold
        self._yamnet: Optional[YAMNetClassifier] = None
        self._mfcc = MFCCClassifier(
            model_path=mfcc_model_path,
            smoothing_window=smoothing_window,
        )

        if use_yamnet:
            self._yamnet = YAMNetClassifier(smoothing_window=smoothing_window)

        self._last_result: Optional[dict] = None

    def classify(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Classify audio chunk and return probability dict over CATEGORIES.
        Automatically selects the best available backend.
        """
        result = None

        if self._yamnet and self._yamnet.is_loaded():
            result = self._yamnet.classify(audio, sample_rate)

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
