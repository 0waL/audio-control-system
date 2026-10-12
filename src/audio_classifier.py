"""
Real-time audio classifier using YAMNet (transfer learning) and custom models.

YAMNet is pre-trained on AudioSet (521 classes). Its class scores are mapped to
our 5 target categories by class name, or — when a head trained with
`training/train_cnn.py --mode yamnet` is supplied — its embeddings are fed to
that learned classification head. A mel-spectrogram CNN (`--mode cnn`) or an
sklearn MFCC model (`--mode mfcc`) can be loaded instead, and an MFCC heuristic
classifier is always available as the final fallback.
"""

import numpy as np
import logging
from collections import deque
from pathlib import Path
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

YAMNET_SAMPLE_RATE = 16000
YAMNET_INPUT_SECONDS = 1.0     # every chunk is fitted to this length
YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"

# YAMNet (AudioSet) display names mapped to our 5 categories. Indices are
# resolved at runtime from the class map shipped with the model.
YAMNET_CATEGORY_CLASSES = {
    "alarm": [
        "Alarm", "Alarm clock", "Siren", "Civil defense siren", "Buzzer",
        "Smoke detector, smoke alarm", "Fire alarm", "Foghorn", "Car alarm",
        "Emergency vehicle", "Police car (siren)", "Ambulance (siren)",
        "Fire engine, fire truck (siren)", "Vehicle horn, car horn, honking",
        "Air horn, truck horn", "Reversing beeps", "Screaming", "Explosion",
        "Gunshot, gunfire", "Shatter",
    ],
    "speech": [
        "Speech", "Child speech, kid speaking", "Conversation",
        "Narration, monologue", "Babbling", "Speech synthesizer", "Shout",
        "Yell", "Whispering", "Laughter", "Crying, sobbing",
        "Baby cry, infant cry",
    ],
    # Plus every class between YAMNET_MUSIC_RANGE (instruments and genres)
    "music": ["Singing", "Choir", "Rapping", "Humming"],
    "keyboard": ["Typing", "Typewriter", "Computer keyboard", "Clicking"],
    "environment": [
        "Wind", "Rustling leaves", "Wind noise (microphone)", "Thunderstorm",
        "Thunder", "Rain", "Rain on surface", "Water", "Stream", "Ocean",
        "Waves, surf", "Bird", "Crowd", "Hubbub, speech noise, speech babble",
        "Vehicle", "Traffic noise, roadway noise", "Engine", "Mechanical fan",
        "Air conditioning", "Vacuum cleaner", "Inside, small room",
        "Inside, large room or hall", "Inside, public space",
        "Outside, urban or manmade", "Outside, rural or natural", "Noise",
        "Environmental noise", "Static", "Mains hum", "Hum", "White noise",
        "Pink noise",
    ],
}
YAMNET_MUSIC_RANGE = ("Music", "Scary music")

# Below this best per-category YAMNet score the chunk is treated as
# "nothing recognisable" (e.g. silence) rather than normalised up.
YAMNET_MIN_SCORE = 0.1


def _uniform_probs() -> dict:
    return {cat: 1.0 / len(CATEGORIES) for cat in CATEGORIES}


def build_category_indices(class_names: list) -> dict:
    """Resolve YAMNET_CATEGORY_CLASSES to YAMNet class indices."""
    lookup = {name: i for i, name in enumerate(class_names)}
    indices = {}
    for cat, names in YAMNET_CATEGORY_CLASSES.items():
        missing = [n for n in names if n not in lookup]
        if missing:
            logger.warning(f"YAMNet classes not found for '{cat}': {missing}")
        indices[cat] = [lookup[n] for n in names if n in lookup]

    first, last = YAMNET_MUSIC_RANGE
    if first in lookup and last in lookup:
        music_range = range(lookup[first], lookup[last] + 1)
        indices["music"] = sorted(set(indices["music"]) | set(music_range))
    else:
        logger.warning(f"YAMNet music range {YAMNET_MUSIC_RANGE} not found")
    return indices


def aggregate_scores(scores: np.ndarray, category_indices: dict) -> dict:
    """
    Collapse YAMNet per-class scores into our categories (max over member
    classes), normalised to sum to 1.
    """
    raw = {
        cat: float(np.max(scores[idx])) if idx else 0.0
        for cat, idx in category_indices.items()
    }
    if max(raw.values()) < YAMNET_MIN_SCORE:
        return _uniform_probs()
    total = sum(raw.values())
    return {cat: raw[cat] / total for cat in CATEGORIES}


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Keep the most recent n samples, or zero-pad up to n."""
    if len(audio) >= n:
        return audio[-n:]
    return np.pad(audio, (0, n - len(audio)))


def _resample(audio: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    if sample_rate == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)


class _TemporalSmoother:
    """Moving average of the last `window` probability dicts."""

    def __init__(self, window: int):
        self._history: deque = deque(maxlen=max(1, window))

    def update(self, probs: dict) -> dict:
        self._history.append(probs)
        return {cat: float(np.mean([h[cat] for h in self._history]))
                for cat in CATEGORIES}


class YAMNetClassifier:
    """
    YAMNet-based classifier. Without a head, class scores are mapped to our
    categories by name; with a trained head (Keras model on 1024-d mean
    embeddings), the head's softmax output is used directly.

    Each chunk is fitted to YAMNET_INPUT_SECONDS of 16 kHz audio. `model_handle`
    is the TF Hub URL or a local path of a downloaded YAMNet SavedModel.
    """

    def __init__(self, smoothing_window: int = 5, head=None,
                 model_handle: str = YAMNET_HUB_URL):
        self._model = None
        self._model_handle = model_handle
        self._head = head
        self._category_indices: dict = {}
        self._loaded = False
        self._smoother = _TemporalSmoother(smoothing_window)
        self._load_model()

    def _load_model(self):
        try:
            import tensorflow_hub as hub
            self._model = hub.load(self._model_handle)
            # Resolve class names from the model asset
            class_map_path = self._model.class_map_path().numpy().decode()
            import csv
            with open(class_map_path, encoding="utf-8", newline="") as f:
                class_names = [row["display_name"] for row in csv.DictReader(f)]
            self._category_indices = build_category_indices(class_names)
            self._loaded = True
            logger.info("YAMNet model loaded successfully"
                        + (" (with trained head)" if self._head is not None else ""))
        except Exception as e:
            logger.warning(f"YAMNet unavailable ({e}), using fallback classifier")
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, audio: np.ndarray, sample_rate: int = YAMNET_SAMPLE_RATE) -> Optional[dict]:
        """
        Returns normalized probability dict over CATEGORIES.
        audio: float32 mono array; resampled to 16 kHz and fitted to
        YAMNET_INPUT_SECONDS.
        """
        if not self._loaded:
            return None

        try:
            import tensorflow as tf
            audio = _resample(audio, sample_rate, YAMNET_SAMPLE_RATE)
            audio = _fit_length(audio, int(YAMNET_SAMPLE_RATE * YAMNET_INPUT_SECONDS))
            # scores: (frames, 521), embeddings: (frames, 1024)
            scores, embeddings, _ = self._model(tf.constant(audio.astype(np.float32)))
            if self._head is not None:
                emb = tf.reduce_mean(embeddings, axis=0)[tf.newaxis, :]
                out = np.asarray(self._head(emb, training=False))[0]
                probs = {cat: float(p) for cat, p in zip(CATEGORIES, out)}
            else:
                mean_scores = tf.reduce_mean(scores, axis=0).numpy()
                probs = aggregate_scores(mean_scores, self._category_indices)
            return self._smoother.update(probs)
        except Exception as e:
            logger.error(f"YAMNet inference error: {e}")
            return None


class CNNClassifier:
    """
    Mel-spectrogram CNN trained by `training/train_cnn.py --mode cnn`.

    Feature extraction mirrors data/preprocessing.py: 16 kHz audio clipped or
    zero-padded to 1 s, log-mel with hop 512. Run with --chunk-ms 1000 to feed
    the model the same clip length it was trained on.
    """

    SAMPLE_RATE = 16000     # data/preprocessing.py SAMPLE_RATE
    CLIP_DURATION = 1.0     # data/preprocessing.py CLIP_DURATION
    HOP_LENGTH = 512        # data/preprocessing.py HOP_LENGTH

    def __init__(self, model, smoothing_window: int = 5):
        self._model = model
        _, self._n_mels, self._frames, _ = model.input_shape
        self._smoother = _TemporalSmoother(smoothing_window)

    def is_loaded(self) -> bool:
        return True

    def _log_mel(self, audio: np.ndarray) -> np.ndarray:
        import librosa
        target = int(self.SAMPLE_RATE * self.CLIP_DURATION)
        if len(audio) >= target:
            audio = audio[:target]
        else:
            audio = np.pad(audio, (0, target - len(audio)))
        mel = librosa.feature.melspectrogram(y=audio, sr=self.SAMPLE_RATE,
                                             n_mels=self._n_mels,
                                             hop_length=self.HOP_LENGTH)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if self._frames is not None:
            if log_mel.shape[1] >= self._frames:
                log_mel = log_mel[:, :self._frames]
            else:
                pad = self._frames - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad)),
                                 constant_values=float(log_mel.min()))
        return log_mel.astype(np.float32)

    def classify(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[dict]:
        try:
            audio = _resample(audio, sample_rate, self.SAMPLE_RATE)
            x = self._log_mel(audio)[np.newaxis, ..., np.newaxis]
            out = np.asarray(self._model(x, training=False))[0]
        except Exception as e:
            logger.error(f"CNN inference error: {e}")
            return None
        probs = {cat: float(p) for cat, p in zip(CATEGORIES, out)}
        return self._smoother.update(probs)


def _load_keras_classifier(path: str, smoothing_window: int):
    """Load a .keras model and wrap it as a CNN or YAMNet-head classifier."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
    except Exception as e:
        logger.warning(f"Could not load Keras model {path} ({e})")
        return None

    if len(model.input_shape) == 2:
        logger.info(f"Loaded YAMNet classification head from {path}")
        return YAMNetClassifier(smoothing_window=smoothing_window, head=model)
    logger.info(f"Loaded mel-spectrogram CNN from {path}")
    return CNNClassifier(model, smoothing_window=smoothing_window)


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
        self._smoother = _TemporalSmoother(smoothing_window)
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
            # Silence — no category stands out
            return _uniform_probs()

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

    def _model_classify(self, audio: np.ndarray, sample_rate: int) -> dict:
        features = self._extract_features(audio, sample_rate)
        probs = self._model.predict_proba([features])[0]
        # classes_ holds the CATEGORIES indices the model was trained on; a
        # category missing from the training data has no column at all.
        classes = getattr(self._model, "classes_", range(len(probs)))
        result = {cat: 0.0 for cat in CATEGORIES}
        for cls, p in zip(classes, probs):
            result[CATEGORIES[int(cls)]] = float(p)
        return result

    def classify(self, audio: np.ndarray, sample_rate: int = 22050) -> dict:
        if self._model is not None:
            try:
                result = self._model_classify(audio, sample_rate)
            except Exception as e:
                logger.warning(f"Model inference failed ({e}), using heuristics")
                result = self._heuristic_classify(audio, sample_rate)
        else:
            result = self._heuristic_classify(audio, sample_rate)

        return self._smoother.update(result)


class AudioClassifier:
    """
    Unified classifier with a single classify() interface.

    Backend order: an explicitly supplied model (.pkl MFCC model, .keras CNN or
    YAMNet head) → YAMNet name mapping (if enabled and no .pkl was given) →
    MFCC heuristics.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        detection_threshold: float = 0.3,
        model_path: Optional[str] = None,
        use_yamnet: bool = True,
    ):
        self._threshold = detection_threshold
        suffix = Path(model_path).suffix.lower() if model_path else ""

        self._mfcc = MFCCClassifier(
            model_path=model_path if suffix == ".pkl" else None,
            smoothing_window=smoothing_window,
        )

        self._primary = None
        if suffix in (".keras", ".h5"):
            self._primary = _load_keras_classifier(model_path, smoothing_window)
        elif model_path and suffix != ".pkl":
            logger.warning(f"Unsupported model file {model_path} (expected .pkl or .keras)")

        if self._primary is None and use_yamnet and suffix != ".pkl":
            self._primary = YAMNetClassifier(smoothing_window=smoothing_window)

        self._last_result: Optional[dict] = None

    def classify(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Classify audio chunk and return probability dict over CATEGORIES.
        Automatically selects the best available backend.
        """
        result = None

        if self._primary is not None and self._primary.is_loaded():
            result = self._primary.classify(audio, sample_rate)

        if result is None:
            result = self._mfcc.classify(audio, sample_rate)

        self._last_result = result
        return result

    def set_threshold(self, threshold: float):
        self._threshold = threshold

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
