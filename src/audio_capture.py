"""
Real-time microphone capture using sounddevice.

Captures audio in overlapping chunks and delivers them to a callback.
Designed to run in a background thread with minimal latency.
"""

import threading
import queue
import logging
import numpy as np
from typing import Callable, Optional

logger = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray, int], None]

# 16 kHz is the project standard (CNN mel features); heuristic works at any rate.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 500
DEFAULT_OVERLAP_MS = 100


class AudioCapture:
    """
    Captures microphone input in real time and feeds chunks to a callback.

    The callback receives (audio_chunk: np.ndarray[float32], sample_rate: int).
    Chunks overlap by `overlap_ms` to avoid classification artifacts at boundaries.
    """

    def __init__(
        self,
        callback: AudioCallback,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        overlap_ms: int = DEFAULT_OVERLAP_MS,
        device: Optional[int] = None,
        channels: int = 1,
    ):
        self._callback = callback
        self._sample_rate = sample_rate
        self._chunk_samples = int(sample_rate * chunk_ms / 1000)
        self._overlap_samples = int(sample_rate * overlap_ms / 1000)
        self._device = device
        self._channels = channels
        # Hardware capture rate; may differ from target rate and get resampled.
        self._capture_rate = sample_rate

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self._buffer = np.zeros(self._chunk_samples, dtype=np.float32)
        self._buffer_pos = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Audio capture started: {self._sample_rate} Hz, "
            f"chunk {self._chunk_samples} samples, "
            f"overlap {self._overlap_samples} samples"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("Audio capture stopped")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @staticmethod
    def list_devices() -> list[dict]:
        """Return list of available audio input devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return [
                {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]
        except Exception as e:
            logger.error(f"Cannot list audio devices: {e}")
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sounddevice_callback(self, indata: np.ndarray, frames: int,
                               time_info, status):
        """sounddevice stream callback — called from audio thread."""
        if status:
            logger.warning(f"sounddevice status: {status}")
        # Downmix to mono (average channels) so multi-channel mics still work.
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
        # Resample to the target rate if the device captured at a different rate.
        if self._capture_rate != self._sample_rate and len(mono) > 1:
            n_out = int(round(len(mono) * self._sample_rate / self._capture_rate))
            if n_out > 0:
                mono = np.interp(
                    np.linspace(0, len(mono), n_out, endpoint=False),
                    np.arange(len(mono)),
                    mono,
                ).astype(np.float32)
        try:
            self._audio_queue.put_nowait(np.ascontiguousarray(mono, dtype=np.float32))
        except queue.Full:
            logger.debug("Audio queue full, dropping frame")

    def _capture_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed; using synthetic audio for testing")
            self._synthetic_loop()
            return

        channels, samplerate = self._resolve_stream_params(sd)
        if channels is None:
            return  # device unusable; error already logged

        self._channels = channels
        self._capture_rate = samplerate
        try:
            with sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                device=self._device,
                callback=self._sounddevice_callback,
                blocksize=int(samplerate * 0.05),  # 50 ms blocks
            ):
                self._process_queue()
        except Exception as e:
            logger.error(
                f"Audio capture error: {e}. "
                f"Check the device index with: python main.py --list-devices"
            )

    def _resolve_stream_params(self, sd):
        """Pick a channel count and sample rate the device actually supports.

        Returns (channels, samplerate), or (None, None) if the device has no
        usable input. Falls back to the device's default sample rate when the
        target rate is unsupported (audio is resampled in the callback).
        """
        channels = self._channels
        samplerate = self._sample_rate
        try:
            info = sd.query_devices(self._device, "input")
        except Exception as e:
            logger.warning(f"Could not query device {self._device}: {e}")
            return channels, samplerate

        name = info.get("name", "?")
        max_in = int(info.get("max_input_channels", 0))
        if max_in <= 0:
            logger.error(
                f"Device {self._device} ('{name}') has no input channels — "
                f"it can't record. Run: python main.py --list-devices "
                f"and pass a microphone index via --device."
            )
            return None, None
        if channels > max_in:
            logger.warning(
                f"Device '{name}' supports {max_in} input channel(s); "
                f"requested {channels}. Using {max_in}."
            )
            channels = max_in

        try:
            sd.check_input_settings(
                device=self._device, channels=channels,
                samplerate=samplerate, dtype="float32",
            )
        except Exception as e:
            fallback = int(info.get("default_samplerate", 0)) or samplerate
            logger.warning(
                f"{samplerate} Hz not supported by '{name}' ({e}); "
                f"capturing at {fallback} Hz and resampling to {self._sample_rate} Hz."
            )
            samplerate = fallback

        return channels, samplerate

    def _process_queue(self):
        """Assemble overlapping chunks from queued raw blocks and call callback."""
        ring = np.zeros(self._chunk_samples * 2, dtype=np.float32)
        write_pos = 0

        while self._running:
            try:
                block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Append block to ring buffer
            n = len(block)
            if write_pos + n > len(ring):
                # Shift ring buffer
                keep = self._chunk_samples - self._overlap_samples
                ring[:keep] = ring[write_pos - keep: write_pos]
                write_pos = keep

            ring[write_pos: write_pos + n] = block
            write_pos += n

            # Emit chunk once enough samples accumulated
            while write_pos >= self._chunk_samples:
                chunk = ring[:self._chunk_samples].copy()
                try:
                    self._callback(chunk, self._sample_rate)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

                # Advance with overlap
                advance = self._chunk_samples - self._overlap_samples
                ring[: write_pos - advance] = ring[advance: write_pos]
                write_pos -= advance

    def _synthetic_loop(self):
        """Generate synthetic audio for unit testing without a microphone."""
        import time
        t = 0.0
        while self._running:
            dt = self._chunk_samples / self._sample_rate
            samples = np.random.randn(self._chunk_samples).astype(np.float32) * 0.01
            # Mix in a 440 Hz sine tone to simulate some signal
            samples += 0.1 * np.sin(
                2 * np.pi * 440 * np.linspace(t, t + dt, self._chunk_samples)
            ).astype(np.float32)
            try:
                self._callback(samples, self._sample_rate)
            except Exception as e:
                logger.error(f"Callback error: {e}")
            t += dt
            time.sleep(dt)
