"""
Audio preprocessing utilities for dataset preparation and feature extraction.

Supports:
  - Loading audio files (WAV, MP3, FLAC) with resampling
  - MFCC + mel-spectrogram feature extraction
  - Dataset loading from UrbanSound8K, ESC-50, and plain per-category folders
  - Train/validation split with stratification
  - Augmentation: time-stretch, pitch-shift, additive noise
"""

import os
import csv
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000       # YAMNet requirement; MFCC model also uses this
CLIP_DURATION = 1.0       # seconds per training sample
N_MFCC = 13
N_MELS = 64
HOP_LENGTH = 512

# Maps dataset-specific class labels to our 5 categories
URBANSOUND_MAP = {
    "air_conditioner": "environment",
    "car_horn": "alarm",
    "children_playing": "environment",
    "dog_bark": "environment",
    "drilling": "environment",
    "engine_idling": "environment",
    "gun_shot": "alarm",
    "jackhammer": "environment",
    "siren": "alarm",
    "street_music": "music",
}

ESC50_MAP = {
    "crying_baby": "speech",
    "sneezing": "speech",
    "clapping": "environment",
    "breathing": "environment",
    "coughing": "speech",
    "footsteps": "environment",
    "laughing": "speech",
    "brushing_teeth": "environment",
    "snoring": "speech",
    "drinking_sipping": "environment",
    "door_wood_knock": "keyboard",
    "mouse_click": "keyboard",
    "keyboard_typing": "keyboard",
    "door_wood_creaks": "environment",
    "can_opening": "environment",
    "washing_machine": "environment",
    "vacuum_cleaner": "environment",
    "clock_alarm": "alarm",
    "clock_tick": "environment",
    "glass_breaking": "alarm",
    "crow": "environment",
    "rooster": "environment",
    "pig": "environment",
    "cow": "environment",
    "frog": "environment",
    "cat": "environment",
    "hen": "environment",
    "insects": "environment",
    "sheep": "environment",
    "dog": "environment",
    "rain": "environment",
    "sea_waves": "environment",
    "crackling_fire": "environment",
    "crickets": "environment",
    "chirping_birds": "environment",
    "water_drops": "environment",
    "wind": "environment",
    "pouring_water": "environment",
    "toilet_flush": "environment",
    "thunderstorm": "environment",
    "helicopter": "environment",
    "chainsaw": "environment",
    "siren": "alarm",
    "car_horn": "alarm",
    "engine": "environment",
    "train": "environment",
    "church_bells": "alarm",
    "airplane": "environment",
    "fireworks": "alarm",
    "hand_saw": "environment",
}

CATEGORIES = ["alarm", "speech", "music", "keyboard", "environment"]

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


# ------------------------------------------------------------------
# Core audio loading
# ------------------------------------------------------------------

def load_audio(path: str, target_sr: int = SAMPLE_RATE,
               duration: Optional[float] = None) -> np.ndarray:
    """Load audio file and return mono float32 array at target_sr."""
    import librosa
    audio, _ = librosa.load(path, sr=target_sr, mono=True, duration=duration)
    return audio.astype(np.float32)


def clip_or_pad(audio: np.ndarray, sr: int = SAMPLE_RATE,
                duration: float = CLIP_DURATION) -> np.ndarray:
    target = int(sr * duration)
    if len(audio) >= target:
        return audio[:target]
    return np.pad(audio, (0, target - len(audio)))


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC) -> np.ndarray:
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                 hop_length=HOP_LENGTH)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std])  # shape: (2*n_mfcc,)


def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE,
                             n_mels: int = N_MELS) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                          hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)   # shape: (n_mels, frames)


def extract_full_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract full feature vector used by the MFCC CNN classifier."""
    import librosa
    features = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr,
                                                  hop_length=HOP_LENGTH)
    features.append(float(np.mean(centroid)))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr,
                                                hop_length=HOP_LENGTH)
    features.append(float(np.mean(rolloff)))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr,
                                                    hop_length=HOP_LENGTH)
    features.append(float(np.mean(bandwidth)))

    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    features.append(float(np.mean(zcr)))

    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    features.append(float(np.mean(rms)))

    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------

def augment(audio: np.ndarray, sr: int = SAMPLE_RATE,
            noise_level: float = 0.005) -> np.ndarray:
    import librosa
    ops = np.random.choice(["noise", "stretch", "pitch", "none"], p=[0.3, 0.3, 0.3, 0.1])
    if ops == "noise":
        audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_level
    elif ops == "stretch":
        rate = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        audio = clip_or_pad(audio, sr)
    elif ops == "pitch":
        steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    return audio


# ------------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------------

def load_urbansound8k(root: str) -> Iterator[tuple[np.ndarray, str]]:
    """
    Yield (audio, category) pairs from UrbanSound8K dataset.
    root: path to the UrbanSound8K directory (contains metadata/ and audio/).
    """
    meta_path = Path(root) / "metadata" / "UrbanSound8K.csv"
    with open(meta_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["class"].lower()
            category = URBANSOUND_MAP.get(class_name)
            if category is None:
                continue
            fold = row["fold"]
            fname = row["slice_file_name"]
            fpath = Path(root) / "audio" / f"fold{fold}" / fname
            if not fpath.exists():
                continue
            try:
                audio = load_audio(str(fpath), duration=CLIP_DURATION)
                audio = clip_or_pad(audio)
                yield audio, category
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")


def load_esc50(root: str) -> Iterator[tuple[np.ndarray, str]]:
    """
    Yield (audio, category) pairs from ESC-50 dataset.
    root: path to the ESC-50-master directory.
    """
    meta_path = Path(root) / "meta" / "esc50.csv"
    with open(meta_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["category"].lower()
            category = ESC50_MAP.get(class_name)
            if category is None:
                continue
            fpath = Path(root) / "audio" / row["filename"]
            if not fpath.exists():
                continue
            try:
                audio = load_audio(str(fpath), duration=CLIP_DURATION)
                audio = clip_or_pad(audio)
                yield audio, category
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")


def load_folder(root: str) -> Iterator[tuple[np.ndarray, str]]:
    """
    Yield (audio, category) pairs from a directory with one sub-folder per
    category, e.g. root/speech/*.wav, root/music/*.mp3.

    Use this to add real speech and music recordings (LibriSpeech, GTZAN,
    your own clips, ...) — UrbanSound8K/ESC-50 contain almost none.
    """
    root_path = Path(root)
    for category in CATEGORIES:
        cat_dir = root_path / category
        if not cat_dir.is_dir():
            continue
        for fpath in sorted(cat_dir.rglob("*")):
            if fpath.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            try:
                audio = load_audio(str(fpath), duration=CLIP_DURATION)
                audio = clip_or_pad(audio)
                yield audio, category
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")


def build_feature_dataset(
    iterators: list,
    augment_data: bool = True,
    augment_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) feature matrices from dataset iterators.
    y is integer-encoded (index into CATEGORIES list).
    """
    X, y = [], []
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

    for iterator in iterators:
        for audio, category in iterator:
            feat = extract_full_features(audio)
            X.append(feat)
            y.append(cat_to_idx[category])
            if augment_data:
                for _ in range(augment_factor):
                    aug = augment(audio)
                    X.append(extract_full_features(aug))
                    y.append(cat_to_idx[category])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_mel_dataset(
    iterators: list,
    augment_data: bool = True,
    augment_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) mel-spectrogram dataset for CNN training."""
    X, y = [], []
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

    for iterator in iterators:
        for audio, category in iterator:
            mel = extract_mel_spectrogram(audio)
            X.append(mel)
            y.append(cat_to_idx[category])
            if augment_data:
                for _ in range(augment_factor):
                    aug = augment(audio)
                    X.append(extract_mel_spectrogram(aug))
                    y.append(cat_to_idx[category])

    X_arr = np.array(X, dtype=np.float32)
    # Add channel dim for CNN: (N, n_mels, frames, 1)
    X_arr = X_arr[..., np.newaxis]
    return X_arr, np.array(y, dtype=np.int64)
