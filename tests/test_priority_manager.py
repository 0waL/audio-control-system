"""
Unit tests for PriorityManager — config loading, mutation, and persistence.
"""

import json
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.priority_manager import PriorityManager

CATEGORIES = ["alarm", "speech", "music", "keyboard", "environment"]


@pytest.fixture
def tmp_config(tmp_path):
    """Return a PriorityManager backed by a temp directory."""
    config_file = tmp_path / "priorities.json"
    return PriorityManager(config_path=config_file)


class TestPriorityManager:
    def test_all_categories_present(self, tmp_config):
        cats = tmp_config.get_categories()
        for cat in CATEGORIES:
            assert cat in cats

    def test_sorted_categories_ascending(self, tmp_config):
        sorted_cats = tmp_config.sorted_categories()
        priorities = [info["priority"] for _, info in sorted_cats]
        assert priorities == sorted(priorities)

    def test_set_priority(self, tmp_config):
        tmp_config.set_priority("music", 1)
        cats = tmp_config.get_categories()
        assert cats["music"]["priority"] == 1

    def test_set_invalid_category_raises(self, tmp_config):
        with pytest.raises(ValueError):
            tmp_config.set_priority("nonexistent", 1)

    def test_set_volume_multiplier(self, tmp_config):
        tmp_config.set_volume_multiplier("speech", 0.5)
        cats = tmp_config.get_categories()
        assert abs(cats["speech"]["volume_multiplier"] - 0.5) < 1e-9

    def test_set_multiplier_out_of_range_raises(self, tmp_config):
        with pytest.raises(ValueError):
            tmp_config.set_volume_multiplier("speech", 2.0)

    def test_swap_priorities(self, tmp_config):
        orig_alarm = tmp_config.get_categories()["alarm"]["priority"]
        orig_music = tmp_config.get_categories()["music"]["priority"]
        tmp_config.swap_priorities("alarm", "music")
        cats = tmp_config.get_categories()
        assert cats["alarm"]["priority"] == orig_music
        assert cats["music"]["priority"] == orig_alarm

    def test_config_persisted(self, tmp_path):
        path = tmp_path / "p.json"
        pm = PriorityManager(config_path=path)
        pm.set_priority("music", 1)
        pm2 = PriorityManager(config_path=path)
        assert pm2.get_categories()["music"]["priority"] == 1

    def test_reset_to_defaults(self, tmp_config):
        tmp_config.set_priority("music", 1)
        tmp_config.reset_to_defaults()
        # After reset, alarm should be priority 1 (default)
        cats = tmp_config.get_categories()
        assert cats["alarm"]["priority"] == 1

    def test_get_priority_config_keys(self, tmp_config):
        cfg = tmp_config.get_priority_config()
        for cat in CATEGORIES:
            assert cat in cfg
            assert "priority" in cfg[cat]
            assert "volume_multiplier" in cfg[cat]

    def test_update_setting(self, tmp_config):
        tmp_config.update_setting("base_volume", 0.75)
        assert abs(tmp_config.get_settings()["base_volume"] - 0.75) < 1e-9

    def test_deferred_save(self, tmp_path):
        path = tmp_path / "p.json"
        pm = PriorityManager(config_path=path)
        pm.set_volume_multiplier("speech", 0.9, save=False)
        pm.update_setting("base_volume", 0.6, save=False)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["sound_categories"]["speech"]["volume_multiplier"] != 0.9
        pm.save()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["sound_categories"]["speech"]["volume_multiplier"] == 0.9
        assert on_disk["settings"]["base_volume"] == 0.6
