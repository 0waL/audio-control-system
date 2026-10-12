"""
Unit tests for AudioCapture — validates callback delivery using synthetic loop.

These tests mock out sounddevice to avoid requiring physical hardware.
"""

import time
import threading
import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_capture import AudioCapture


class TestAudioCapture:
    def test_synthetic_loop_delivers_chunks(self):
        received = []
        lock = threading.Event()

        def cb(audio, sr):
            received.append((audio, sr))
            if len(received) >= 3:
                lock.set()

        # Force synthetic loop by making sounddevice unavailable
        with patch.dict("sys.modules", {"sounddevice": None}):
            cap = AudioCapture(callback=cb, sample_rate=16000, chunk_ms=100)
            cap.start()
            lock.wait(timeout=5.0)
            cap.stop()

        assert len(received) >= 3
        for audio, sr in received:
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert sr == 16000

    def test_chunk_length_matches_config(self):
        received = []
        lock = threading.Event()
        sample_rate = 16000
        chunk_ms = 100
        expected_samples = int(sample_rate * chunk_ms / 1000)

        def cb(audio, sr):
            received.append(len(audio))
            if len(received) >= 2:
                lock.set()

        with patch.dict("sys.modules", {"sounddevice": None}):
            cap = AudioCapture(callback=cb, sample_rate=sample_rate,
                               chunk_ms=chunk_ms)
            cap.start()
            lock.wait(timeout=5.0)
            cap.stop()

        assert all(n == expected_samples for n in received)

    def test_list_devices_returns_list(self):
        # Should return a list (may be empty in CI)
        devices = AudioCapture.list_devices()
        assert isinstance(devices, list)

    def test_stop_is_idempotent(self):
        def cb(a, s): pass
        with patch.dict("sys.modules", {"sounddevice": None}):
            cap = AudioCapture(callback=cb)
            cap.start()
            cap.stop()
            cap.stop()  # second stop should not raise

    def test_callback_exception_does_not_crash_capture(self):
        call_count = [0]
        lock = threading.Event()

        def bad_cb(audio, sr):
            call_count[0] += 1
            if call_count[0] >= 3:
                lock.set()
            raise RuntimeError("intentional test error")

        with patch.dict("sys.modules", {"sounddevice": None}):
            cap = AudioCapture(callback=bad_cb, sample_rate=16000, chunk_ms=100)
            cap.start()
            lock.wait(timeout=5.0)
            cap.stop()

        assert call_count[0] >= 3

    def test_default_window_is_one_second_every_400ms(self):
        cap = AudioCapture(callback=lambda a, s: None, sample_rate=16000)
        assert cap._chunk_samples == 16000
        assert cap._chunk_samples - cap._overlap_samples == 6400   # 400 ms hop

    def test_shorter_window_keeps_400ms_hop(self):
        cap = AudioCapture(callback=lambda a, s: None, sample_rate=16000, chunk_ms=500)
        assert cap._chunk_samples - cap._overlap_samples == 6400

    def test_overlap_must_be_shorter_than_chunk(self):
        with pytest.raises(ValueError):
            AudioCapture(callback=lambda a, s: None, chunk_ms=500, overlap_ms=500)
