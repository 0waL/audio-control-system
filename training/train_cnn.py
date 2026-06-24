"""
Audio classification model training.

Modes:
  1. --mode mfcc : Train an sklearn RandomForest on MFCC features
                   (fast, no GPU, works on any Python including 3.14).
  2. --mode cnn  : Train a PyTorch CNN on log-mel spectrograms
                   (the project's deep-learning model; PyTorch has 3.14 wheels).
  3. --mode eval : Evaluate a saved model (.pkl sklearn or .pt PyTorch CNN).

Usage:
  python training/train_cnn.py --mode mfcc --urbansound /data/UrbanSound8K --esc50 /data/ESC-50-master
  python training/train_cnn.py --mode cnn  --urbansound /data/UrbanSound8K --epochs 50
  python training/train_cnn.py --mode eval --model models/cnn_torch.pt --esc50 /data/ESC-50-master
"""

import argparse
import logging
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
# MFCC + RandomForest (sklearn) — works on any Python, incl. 3.14
# ------------------------------------------------------------------

def train_mfcc(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
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
# CNN on log-mel spectrograms (PyTorch)
# ------------------------------------------------------------------

def _prepare_mel_dataset(args):
    """Build the mel dataset and reshape to PyTorch layout (N, 1, n_mels, frames)."""
    iters = _make_iterators(args)
    X, y = build_mel_dataset(iters, augment_data=True, augment_factor=2)
    # build_mel_dataset returns (N, n_mels, frames, 1) → (N, 1, n_mels, frames)
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
    return X, y


def train_cnn(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from src.cnn_model import AudioCNN

    logger.info("Building log-mel spectrogram dataset ...")
    X, y = _prepare_mel_dataset(args)
    logger.info(f"Dataset: {X.shape} (N, C, n_mels, frames)")

    n_mels = X.shape[2]
    n_classes = len(CATEGORIES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Normalise log-mel values with train-set stats; store for inference.
    mean = float(X_train.mean())
    std = float(X_train.std()) or 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test),
                            torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = AudioCNN(n_mels=n_mels, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_acc = 0.0
    best_state = None
    patience, bad_epochs = 8, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_ds)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
        val_acc = correct / len(test_ds)
        scheduler.step(val_acc)
        logger.info(f"Epoch {epoch:3d} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best_acc {best_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final report
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(yb.numpy())
    print(classification_report(all_true, all_pred, target_names=CATEGORIES))
    print("Confusion matrix:")
    print(confusion_matrix(all_true, all_pred))
    logger.info(f"Best validation accuracy: {best_acc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "cnn_torch.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "categories": CATEGORIES,
        "n_mels": n_mels,
        "mean": mean,
        "std": std,
    }, model_path)
    logger.info(f"PyTorch CNN saved to {model_path}")
    return str(model_path)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(args):
    from sklearn.metrics import classification_report, confusion_matrix

    iters = _make_iterators(args)
    logger.info("Evaluating model ...")

    if args.model.endswith(".pkl"):
        import pickle
        X, y = build_feature_dataset(iters, augment_data=False)
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
    else:
        import torch
        from src.cnn_model import AudioCNN
        X, y = _prepare_mel_dataset_noaug(iters)
        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
        mean, std = ckpt.get("mean", 0.0), ckpt.get("std", 1.0)
        X = (X - mean) / (std + 1e-6)
        model = AudioCNN(n_mels=ckpt.get("n_mels", X.shape[2]),
                         n_classes=len(ckpt.get("categories", CATEGORIES)))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.from_numpy(X)).argmax(1).numpy()

    print(classification_report(y, y_pred, target_names=CATEGORIES))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))


def _prepare_mel_dataset_noaug(iters):
    X, y = build_mel_dataset(iters, augment_data=False)
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
    return X, y


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
    parser.add_argument("--mode", choices=["mfcc", "cnn", "eval"], default="cnn")
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
    elif args.mode == "eval":
        if not args.model:
            parser.error("--model required for eval mode")
        evaluate(args)


if __name__ == "__main__":
    main()
