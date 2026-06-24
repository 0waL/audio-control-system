"""
PyTorch CNN model for audio classification.

A compact 2-D CNN over log-mel spectrograms, shared by both the training script
(`training/train_cnn.py`) and the real-time inference backend
(`src.audio_classifier.TorchCNNClassifier`) so the architecture stays in one
place. `AdaptiveAvgPool2d` makes the model robust to variable-length input, so
the same weights work on 1.0 s training clips and ~0.5 s real-time chunks.

PyTorch provides wheels for Python 3.14 (unlike TensorFlow), so this runs on the
user's existing interpreter with no separate environment.
"""

import torch
import torch.nn as nn

N_MELS = 64
CATEGORIES = ["alarm", "speech", "music", "keyboard", "environment"]


class AudioCNN(nn.Module):
    """3-block CNN → global average pool → 2-layer classifier head."""

    def __init__(self, n_mels: int = N_MELS, n_classes: int = len(CATEGORIES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, frames)
        x = self.features(x)
        return self.classifier(x)
