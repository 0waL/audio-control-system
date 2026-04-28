"""
Core control engine: connects AudioCapture → AudioClassifier → VolumeController.

Runs the real-time pipeline in background threads and exposes a simple
start/stop/status API consumed by both the UI and the CLI entry point.
"""

import logging
import threading
import time
import numpy as np
from typing import Optional

from src.audio_capture import AudioCapture
from src.audio_classifier import AudioClassifier
from src.volume_controller import VolumeController
from src.priority_manager import PriorityManager

logger = logging.getLogger(__name__)


class AudioControlEngine:
    """
    Orchestrates the full real-time audio pipeline.

    Thread model:
      - AudioCapture runs its own daemon thread for microphone I/O.
      - Each audio chunk is dispatched to the classifier inline in that thread.
      - VolumeController runs its own 50 ms smoothing loop thread.
    """

    def __init__(
        self,
        priority_manager: PriorityManager,
        sample_rate: int = 16000,
        chunk_ms: int = 500,
        overlap_ms: int = 100,
        mic_device: Optional[int] = None,
        mfcc_model_path: Optional[str] = None,
        use_yamnet: bool = True,
    ):
        self._pm = priority_manager
        settings = priority_manager.get_settings()

        self._classifier = AudioClassifier(
            smoothing_window=settings.get("smoothing_window", 5),
            detection_threshold=settings.get("detection_threshold", 0.3),
            mfcc_model_path=mfcc_model_path,
            use_yamnet=use_yamnet,
        )

        self._volume_ctrl = VolumeController(
            base_volume=settings.get("base_volume", 1.0),
            response_time_ms=settings.get("response_time_ms", 200),
            recovery_time_ms=settings.get("recovery_time_ms", 1500),
        )

        self._capture = AudioCapture(
            callback=self._on_audio_chunk,
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
            overlap_ms=overlap_ms,
            device=mic_device,
        )

        self._lock = threading.Lock()
        self._last_probs: dict = {}
        self._last_top_cat: str = "--"
        self._last_confidence: float = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._capture.start()
        logger.info("AudioControlEngine started")

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._capture.stop()
        self._volume_ctrl.stop()
        logger.info("AudioControlEngine stopped")

    def get_status(self) -> dict:
        with self._lock:
            return {
                "top_category": self._last_top_cat,
                "confidence": self._last_confidence,
                "volume": self._volume_ctrl.get_current_volume(),
                "probabilities": dict(self._last_probs),
            }

    def set_base_volume(self, volume: float):
        self._volume_ctrl.set_base_volume(volume)

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------

    def _on_audio_chunk(self, audio: np.ndarray, sample_rate: int):
        """Called from the AudioCapture thread for each new chunk."""
        probs = self._classifier.classify(audio, sample_rate)

        if not self._classifier.is_significant(probs):
            return

        top_cat, confidence = self._classifier.top_category(probs)
        priority_config = self._pm.get_priority_config()
        self._volume_ctrl.apply_priority(probs, priority_config)

        with self._lock:
            self._last_probs = probs
            self._last_top_cat = top_cat
            self._last_confidence = confidence

        logger.debug(
            f"[{top_cat}] {confidence:.2f} | "
            + " ".join(f"{k}:{v:.2f}" for k, v in sorted(probs.items()))
        )
