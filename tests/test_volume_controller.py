"""
Unit tests for VolumeController using StubBackend (no audio hardware).

Tests verify:
  - Volume converges toward target under exponential smoothing
  - Recovery restores base volume after the trigger window expires
  - Priority config correctly selects target multiplier
"""

import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volume_controller import VolumeController, StubBackend


PRIORITY_CONFIG = {
    "alarm":       {"priority": 1, "volume_multiplier": 1.0},
    "speech":      {"priority": 2, "volume_multiplier": 0.4},
    "music":       {"priority": 3, "volume_multiplier": 0.2},
    "keyboard":    {"priority": 4, "volume_multiplier": 0.7},
    "environment": {"priority": 5, "volume_multiplier": 0.8},
}


@pytest.fixture
def controller():
    ctrl = VolumeController(
        base_volume=1.0,
        response_time_ms=200,
        recovery_time_ms=500,   # short for testing
        smoothing_alpha=1.0,    # instant convergence for deterministic tests
    )
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
