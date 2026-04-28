"""
Cross-platform system volume controller.

Priority: PulseAudio (Linux) → ALSA amixer (Linux) → macOS osascript → stub.

Volume changes are smoothed with an exponential moving average to avoid
jarring sudden jumps, and a recovery timer gradually restores the volume
to its baseline after the triggering sound subsides.
"""

import subprocess
import threading
import time
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class VolumeBackend:
    """Abstract volume control backend."""

    def get_volume(self) -> float:
        raise NotImplementedError

    def set_volume(self, level: float):
        raise NotImplementedError


class PulseAudioBackend(VolumeBackend):
    """PulseAudio volume control via pulsectl."""

    def __init__(self, sink_index: Optional[int] = None):
        import pulsectl
        self._pulse = pulsectl.Pulse("audio-priority-control")
        self._sink_index = sink_index

    def _get_sink(self):
        import pulsectl
        sinks = self._pulse.sink_list()
        if not sinks:
            raise RuntimeError("No PulseAudio sinks found")
        if self._sink_index is not None:
            return next(s for s in sinks if s.index == self._sink_index)
        return sinks[0]

    def get_volume(self) -> float:
        sink = self._get_sink()
        return float(sink.volume.value_flat)

    def set_volume(self, level: float):
        level = max(0.0, min(1.5, level))
        sink = self._get_sink()
        self._pulse.volume_set_all_chans(sink, level)


class ALSABackend(VolumeBackend):
    """ALSA volume control via amixer command."""

    def get_volume(self) -> float:
        try:
            out = subprocess.check_output(
                ["amixer", "get", "Master"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "%" in line:
                    pct = int(line.split("[")[1].split("%")[0])
                    return pct / 100.0
        except Exception:
            pass
        return 0.5

    def set_volume(self, level: float):
        pct = int(max(0, min(100, level * 100)))
        subprocess.run(
            ["amixer", "set", "Master", f"{pct}%"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class MacOSBackend(VolumeBackend):
    """macOS volume control via osascript."""

    def get_volume(self) -> float:
        try:
            out = subprocess.check_output(
                ["osascript", "-e", "output volume of (get volume settings)"],
                text=True,
            )
            return int(out.strip()) / 100.0
        except Exception:
            return 0.5

    def set_volume(self, level: float):
        pct = int(max(0, min(100, level * 100)))
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {pct}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class StubBackend(VolumeBackend):
    """In-memory stub for testing without audio hardware."""

    def __init__(self):
        self._volume = 1.0

    def get_volume(self) -> float:
        return self._volume

    def set_volume(self, level: float):
        self._volume = max(0.0, min(1.5, level))
        logger.debug(f"[StubBackend] Volume set to {self._volume:.3f}")


def _create_backend() -> VolumeBackend:
    try:
        import pulsectl
        backend = PulseAudioBackend()
        backend.get_volume()  # validate connection
        logger.info("Using PulseAudio backend")
        return backend
    except Exception:
        pass

    import platform
    if platform.system() == "Darwin":
        logger.info("Using macOS backend")
        return MacOSBackend()

    try:
        ALSABackend().get_volume()
        logger.info("Using ALSA backend")
        return ALSABackend()
    except Exception:
        pass

    logger.warning("No audio backend available; using stub")
    return StubBackend()


class VolumeController:
    """
    Smoothly adjusts system volume based on audio classification results.

    Logic:
    1. Receive classification probabilities + priority config from PriorityManager.
    2. Compute target volume multiplier for the highest-priority detected sound.
    3. Apply exponential smoothing toward target volume.
    4. Start a recovery timer: after `recovery_time_ms` of no high-priority
       detections, gradually restore volume to `base_volume`.
    """

    def __init__(
        self,
        base_volume: float = 1.0,
        response_time_ms: int = 200,
        recovery_time_ms: int = 1500,
        smoothing_alpha: float = 0.3,
    ):
        self._backend = _create_backend()
        self._base_volume = base_volume
        self._response_ms = response_time_ms
        self._recovery_ms = recovery_time_ms
        self._alpha = smoothing_alpha

        self._current_volume = self._backend.get_volume()
        self._target_volume = self._current_volume
        self._last_trigger_time: Optional[float] = None
        self._recovery_active = False

        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._smooth_loop, daemon=True)
        self._running = True
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_priority(self, probs: dict, priority_config: dict):
        """
        Compute and set the target volume based on detected sound probabilities
        and the user-configured priority rules.

        priority_config: dict of category -> {priority, volume_multiplier}
        """
        # Find the highest-priority category that exceeds threshold
        threshold = 0.25
        active = {cat: p for cat, p in probs.items() if p >= threshold}

        if not active:
            return  # nothing significant detected

        # Sort by user-defined priority (lower number = higher priority)
        def sort_key(cat):
            cfg = priority_config.get(cat, {})
            return cfg.get("priority", 99)

        top_cat = min(active, key=sort_key)
        cfg = priority_config.get(top_cat, {})
        multiplier = cfg.get("volume_multiplier", 1.0)
        target = self._base_volume * multiplier

        with self._lock:
            self._target_volume = target
            self._last_trigger_time = time.monotonic()
            self._recovery_active = False

        logger.info(
            f"Volume target → {target:.2f} "
            f"(triggered by '{top_cat}', confidence {active[top_cat]:.2f})"
        )

    def set_base_volume(self, volume: float):
        with self._lock:
            self._base_volume = max(0.0, min(1.5, volume))

    def get_current_volume(self) -> float:
        return self._current_volume

    def stop(self):
        self._running = False
        self._worker.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal smoothing loop
    # ------------------------------------------------------------------

    def _smooth_loop(self):
        interval = 0.05  # 50 ms update tick

        while self._running:
            time.sleep(interval)

            with self._lock:
                target = self._target_volume
                last_trigger = self._last_trigger_time

            # Check recovery: if enough time passed since last trigger, ramp back
            now = time.monotonic()
            if last_trigger is not None:
                elapsed_ms = (now - last_trigger) * 1000
                if elapsed_ms > self._recovery_ms and not self._recovery_active:
                    with self._lock:
                        self._target_volume = self._base_volume
                        self._recovery_active = True
                    logger.debug("Volume recovery: restoring base volume")

            # Exponential moving average toward target
            new_vol = self._alpha * target + (1.0 - self._alpha) * self._current_volume
            if abs(new_vol - self._current_volume) > 0.001:
                self._current_volume = new_vol
                try:
                    self._backend.set_volume(new_vol)
                except Exception as e:
                    logger.error(f"Volume set error: {e}")
