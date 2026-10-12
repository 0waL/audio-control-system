"""
Unit tests for VolumeController using StubBackend (no audio hardware).

Tests verify:
  - Volume converges toward target under exponential smoothing
  - Recovery restores base volume after the trigger window expires
  - Priority config correctly selects target multiplier
  - Volume is relative to, and restored to, the level found at start()
  - start/stop can be repeated; settings apply while running
"""

import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

from src.volume_controller import VolumeController, StubBackend, ALSABackend


PRIORITY_CONFIG = {
    "alarm":       {"priority": 1, "volume_multiplier": 1.0},
    "speech":      {"priority": 2, "volume_multiplier": 0.4},
    "music":       {"priority": 3, "volume_multiplier": 0.2},
    "keyboard":    {"priority": 4, "volume_multiplier": 0.7},
    "environment": {"priority": 5, "volume_multiplier": 0.8},
}


SPEECH = {"alarm": 0.0, "speech": 0.9, "music": 0.0,
          "keyboard": 0.0, "environment": 0.0}


def _make(system_volume=1.0, **kwargs):
    backend = StubBackend()
    backend.set_volume(system_volume)
    params = dict(base_volume=1.0, response_time_ms=200,
                  recovery_time_ms=500,   # short for testing
                  smoothing_alpha=1.0,    # instant convergence for deterministic tests
                  backend=backend)
    params.update(kwargs)
    return VolumeController(**params), backend


@pytest.fixture
def controller():
    ctrl, _ = _make()
    ctrl.start()
    yield ctrl
    ctrl.stop()


class TestVolumeController:
    def test_initial_volume_non_negative(self, controller):
        assert controller.get_current_volume() >= 0.0

    def test_apply_speech_reduces_volume(self, controller):
        probs = {"alarm": 0.0, "speech": 0.9, "music": 0.0,
                 "keyboard": 0.0, "environment": 0.0}
        controller.apply_priority(probs, PRIORITY_CONFIG)
        time.sleep(0.15)
        # With alpha=1.0, volume should converge to base * 0.4 = 0.4
        assert controller.get_current_volume() < 0.8

    def test_alarm_keeps_full_volume(self, controller):
        probs = {"alarm": 0.9, "speech": 0.05, "music": 0.0,
                 "keyboard": 0.0, "environment": 0.05}
        controller.apply_priority(probs, PRIORITY_CONFIG)
        time.sleep(0.15)
        # alarm multiplier = 1.0, so volume stays at base (1.0)
        vol = controller.get_current_volume()
        assert vol >= 0.8

    def test_higher_priority_wins_over_lower(self, controller):
        # Both speech (priority 2) and music (priority 3) above threshold
        probs = {"alarm": 0.0, "speech": 0.4, "music": 0.4,
                 "keyboard": 0.1, "environment": 0.1}
        controller.apply_priority(probs, PRIORITY_CONFIG)
        time.sleep(0.15)
        # Speech wins (priority 2 < 3); multiplier = 0.4
        assert controller.get_current_volume() < 0.8

    def test_set_base_volume(self, controller):
        controller.set_base_volume(0.5)
        # After recovery the volume should tend toward 0.5
        time.sleep(0.8)
        vol = controller.get_current_volume()
        assert vol <= 1.0

    def test_below_threshold_no_change(self, controller):
        # All probabilities below 0.25 threshold → no trigger
        initial = controller.get_current_volume()
        probs = {cat: 0.04 for cat in PRIORITY_CONFIG}
        controller.apply_priority(probs, PRIORITY_CONFIG)
        time.sleep(0.1)
        # Volume should not have changed significantly
        assert abs(controller.get_current_volume() - initial) < 0.5

    def test_recovery_returns_to_starting_volume_not_full(self):
        ctrl, backend = _make(system_volume=0.6)
        ctrl.start()
        try:
            ctrl.apply_priority(SPEECH, PRIORITY_CONFIG)
            time.sleep(0.15)
            assert abs(backend.get_volume() - 0.6 * 0.4) < 0.01
            time.sleep(0.7)   # past recovery_time_ms
            assert abs(backend.get_volume() - 0.6) < 0.01
        finally:
            ctrl.stop()

    def test_stop_restores_starting_volume(self):
        ctrl, backend = _make(system_volume=0.7)
        ctrl.start()
        ctrl.apply_priority(SPEECH, PRIORITY_CONFIG)
        time.sleep(0.15)
        assert backend.get_volume() < 0.5
        ctrl.stop()
        assert abs(backend.get_volume() - 0.7) < 1e-9

    def test_restart_after_stop(self):
        ctrl, backend = _make()
        ctrl.start()
        ctrl.stop()
        ctrl.start()
        try:
            ctrl.apply_priority(SPEECH, PRIORITY_CONFIG)
            time.sleep(0.15)
            assert backend.get_volume() < 0.8
        finally:
            ctrl.stop()

    def test_detection_threshold_is_configurable(self):
        ctrl, backend = _make(detection_threshold=0.5)
        ctrl.start()
        try:
            probs = dict(SPEECH, speech=0.45)
            ctrl.apply_priority(probs, PRIORITY_CONFIG)
            time.sleep(0.15)
            assert backend.get_volume() > 0.9
            ctrl.set_detection_threshold(0.4)
            ctrl.apply_priority(probs, PRIORITY_CONFIG)
            time.sleep(0.15)
            assert backend.get_volume() < 0.8
        finally:
            ctrl.stop()

    def test_faint_trigger_keeps_holding(self):
        ctrl, backend = _make(recovery_time_ms=300)
        ctrl.start()
        try:
            ctrl.apply_priority(SPEECH, PRIORITY_CONFIG)
            # Speech now below the 0.3 threshold but still dominant
            faint = {"alarm": 0.1, "speech": 0.25, "music": 0.2,
                     "keyboard": 0.2, "environment": 0.25 - 0.01}
            for _ in range(8):
                time.sleep(0.1)
                ctrl.apply_priority(faint, PRIORITY_CONFIG)
            assert backend.get_volume() < 0.8
        finally:
            ctrl.stop()

    def test_flat_result_releases_hold(self):
        ctrl, backend = _make(recovery_time_ms=300)
        ctrl.start()
        try:
            ctrl.apply_priority(SPEECH, PRIORITY_CONFIG)
            flat = {cat: 0.2 for cat in PRIORITY_CONFIG}   # e.g. silence
            for _ in range(8):
                time.sleep(0.1)
                ctrl.apply_priority(flat, PRIORITY_CONFIG)
            assert backend.get_volume() > 0.9
        finally:
            ctrl.stop()

    def test_response_time_sets_smoothing(self):
        slow, _ = _make(response_time_ms=1000, smoothing_alpha=None)
        fast, _ = _make(response_time_ms=100, smoothing_alpha=None)
        assert 0.0 < slow._alpha < fast._alpha < 1.0
        slow.set_timing(100, 500)
        assert abs(slow._alpha - fast._alpha) < 1e-12


class TestBackendDetection:
    def test_alsa_unavailable_without_amixer(self):
        with patch("src.volume_controller.shutil.which", return_value=None):
            assert ALSABackend.is_available() is False
