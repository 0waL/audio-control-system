"""
Tests for AudioControlEngine wiring and the CLI loop (no audio hardware).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import main
from src.engine import AudioControlEngine
from src.priority_manager import PriorityManager
from src.volume_controller import StubBackend


@pytest.fixture
def engine(tmp_path):
    pm = PriorityManager(config_path=tmp_path / "p.json")
    with patch("src.volume_controller._create_backend", return_value=StubBackend()):
        eng = AudioControlEngine(pm, chunk_ms=100, use_yamnet=False)
    yield eng
    eng.stop()


def test_engine_can_restart(engine):
    with patch.dict("sys.modules", {"sounddevice": None}):
        engine.start()
        engine.stop()
        engine.start()
        assert engine._volume_ctrl._worker.is_alive()
        assert engine._capture._thread.is_alive()
        engine.stop()


def test_update_settings_reaches_components(engine):
    engine.update_settings({
        "detection_threshold": 0.6, "base_volume": 0.5,
        "response_time_ms": 100, "recovery_time_ms": 900,
    })
    assert not engine._classifier.is_significant({"speech": 0.55})
    assert engine._volume_ctrl._threshold == 0.6
    assert engine._volume_ctrl._base_volume == 0.5
    assert engine._volume_ctrl._recovery_ms == 900


def test_cli_loop_runs_and_stops_engine():
    eng = MagicMock()
    eng.get_status.return_value = {
        "top_category": "speech", "confidence": 0.5, "volume": 0.4,
        "probabilities": {"speech": 0.5},
    }
    fake_time = SimpleNamespace(sleep=MagicMock(side_effect=KeyboardInterrupt))
    fake_signal = SimpleNamespace(signal=MagicMock(), SIGINT=2)
    with patch.object(main, "time", fake_time), \
            patch.object(main, "signal", fake_signal):
        with pytest.raises(KeyboardInterrupt):
            main._run_cli(eng, None)
    eng.start.assert_called_once()
    eng.stop.assert_called_once()
