"""
CNN model training for audio classification.

Two modes:
  1. --mode mfcc   : Train an sklearn RandomForest on MFCC features (fast, no GPU needed)
  2. --mode cnn    : Train a TensorFlow CNN on mel-spectrograms (higher accuracy)
  3. --mode yamnet : Fine-tune a classification head on top of YAMNet embeddings

Usage:
  python training/train_cnn.py --mode mfcc --urbansound /data/UrbanSound8K --esc50 /data/ESC-50
  python training/train_cnn.py --mode cnn  --urbansound /data/UrbanSound8K --epochs 50
"""

import argparse
import logging
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import (
    load_urbansound8k, load_esc50,
    build_feature_dataset, build_mel_dataset,
    CATEGORIES, SAMPLE_RATE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"


# ------------------------------------------------------------------
# MFCC + RandomForest (sklearn)
# ------------------------------------------------------------------

def train_mfcc(args):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, f1_score
    from sklearn.pipeline import Pipeline
    import pickle

    logger.info("Building MFCC feature dataset ...")
    iters = _make_iterators(args)
    X, y = build_feature_dataset(iters, augment_data=True, augment_factor=2)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
        )),
    ])

    logger.info("Training RandomForest ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=CATEGORIES)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info(f"\n{report}")
    logger.info(f"Weighted F1: {f1:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "mfcc_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved to {model_path}")
    return str(model_path)


# ------------------------------------------------------------------
# CNN on mel-spectrograms (TensorFlow/Keras)
# ------------------------------------------------------------------

def train_cnn(args):
    import tensorflow as tf
    from tensorflow import keras

    logger.info("Building mel-spectrogram dataset ...")
    iters = _make_iterators(args)
    X, y = build_mel_dataset(iters, augment_data=True, augment_factor=2)
    logger.info(f"Dataset: {X.shape} mel-spectrograms")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    n_mels, frames, _ = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    n_classes = len(CATEGORIES)

    model = _build_cnn(n_mels, frames, n_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "cnn_best.keras"), save_best_only=True
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=32,
        callbacks=callbacks,
    )

    results = model.evaluate(X_test, y_test)
    logger.info(f"Test loss: {results[0]:.4f}, accuracy: {results[1]:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "cnn_classifier.keras"
    model.save(str(model_path))
    logger.info(f"CNN model saved to {model_path}")
    return str(model_path)


def _build_cnn(n_mels: int, frames: int, n_classes: int):
    from tensorflow import keras

    inputs = keras.Input(shape=(n_mels, frames, 1))

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------------------------------------------------
# YAMNet fine-tuning (classification head only)
# ------------------------------------------------------------------

def train_yamnet_head(args):
    """
    Extract YAMNet embeddings for all training samples, then train a
    dense classification head (much faster than full fine-tuning).
    """
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow import keras
    from sklearn.model_selection import train_test_split

    logger.info("Loading YAMNet ...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    logger.info("Extracting YAMNet embeddings ...")
    iters = _make_iterators(args)
    embeddings, labels = [], []
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

    for iterator in iters:
        for audio, category in iterator:
            audio_tf = tf.constant(audio)
            _, emb, _ = yamnet(audio_tf)
            # emb shape: (frames, 1024) — take mean over time
            embeddings.append(tf.reduce_mean(emb, axis=0).numpy())
            labels.append(cat_to_idx[category])

    X = np.array(embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    logger.info(f"Embeddings: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    head = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(1024,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(CATEGORIES), activation="softmax"),
    ])
    head.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    head.fit(X_train, y_train, validation_data=(X_test, y_test),
             epochs=args.epochs, batch_size=64)

    results = head.evaluate(X_test, y_test)
    logger.info(f"YAMNet head — loss: {results[0]:.4f}, acc: {results[1]:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "yamnet_head.keras"
    head.save(str(path))
    logger.info(f"YAMNet head saved to {path}")
    return str(path)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(args):
    from sklearn.metrics import classification_report, confusion_matrix
    import pickle, tensorflow as tf

    iters = _make_iterators(args)
    logger.info("Evaluating model ...")

    if args.model.endswith(".pkl"):
        X, y = build_feature_dataset(iters, augment_data=False)
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
    else:
        X, y = build_mel_dataset(iters, augment_data=False)
        model = tf.keras.models.load_model(args.model)
        y_pred = np.argmax(model.predict(X), axis=1)

    print(classification_report(y, y_pred, target_names=CATEGORIES))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_iterators(args) -> list:
    iters = []
    if args.urbansound:
        iters.append(load_urbansound8k(args.urbansound))
    if args.esc50:
        iters.append(load_esc50(args.esc50))
    if not iters:
        logger.warning("No dataset path provided; generating synthetic data")
        iters.append(_synthetic_iterator(n=500))
    return iters


def _synthetic_iterator(n: int = 200):
    """Generate synthetic labeled samples for smoke-testing without real data."""
    import random
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    for _ in range(n):
        cat = random.choice(CATEGORIES)
        sr = SAMPLE_RATE
        duration = 1.0
        samples = int(sr * duration)
        if cat == "alarm":
            freq = random.uniform(800, 4000)
            t = np.linspace(0, duration, samples)
            audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        elif cat == "speech":
            # Bandpass-ish noise to mimic formant structure
            noise = np.random.randn(samples).astype(np.float32) * 0.1
            audio = noise
        elif cat == "music":
            freqs = [220, 330, 440, 550]
            t = np.linspace(0, duration, samples)
            audio = sum(0.1 * np.sin(2 * np.pi * f * t) for f in freqs)
            audio = audio.astype(np.float32)
        elif cat == "keyboard":
            # Short transients with decay
            audio = np.zeros(samples, dtype=np.float32)
            for pos in range(0, samples, sr // 4):
                burst_len = min(sr // 20, samples - pos)
                burst = np.random.randn(burst_len).astype(np.float32) * 0.3
                burst *= np.exp(-np.linspace(0, 10, burst_len))
                audio[pos:pos + burst_len] += burst
        else:  # environment
            audio = np.random.randn(samples).astype(np.float32) * 0.02

        yield audio, cat


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train audio classification model")
    parser.add_argument("--mode", choices=["mfcc", "cnn", "yamnet", "eval"],
                        default="mfcc")
    parser.add_argument("--urbansound", type=str, default=None,
                        help="Path to UrbanSound8K directory")
    parser.add_argument("--esc50", type=str, default=None,
                        help="Path to ESC-50-master directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (for --mode eval)")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "mfcc":
        train_mfcc(args)
    elif args.mode == "cnn":
        train_cnn(args)
    elif args.mode == "yamnet":
        train_yamnet_head(args)
    elif args.mode == "eval":
        if not args.model:
            parser.error("--model required for eval mode")
        evaluate(args)


if __name__ == "__main__":
    main()
