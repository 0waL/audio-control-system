"""
Priority manager: loads, saves, and exposes user-configured sound priorities.

Priority rules determine which detected sound category triggers a volume change
and by how much. Users can reorder categories and set per-category volume
multipliers (0.0 = mute, 1.0 = no change, >1.0 = boost).
"""

import json
import os
import copy
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default_priorities.json"
USER_CONFIG_PATH = Path.home() / ".audio-control" / "priorities.json"

CATEGORIES = ["alarm", "speech", "music", "keyboard", "environment"]


class PriorityManager:
    """
    Manages sound category priorities and per-category volume multipliers.

    Config is persisted to ~/.audio-control/priorities.json. Falls back to
    the bundled default_priorities.json if user config does not exist.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._path = Path(config_path) if config_path else USER_CONFIG_PATH
        self._config: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_categories(self) -> dict:
        """Return the full sound_categories config dict."""
        return copy.deepcopy(self._config["sound_categories"])

    def get_settings(self) -> dict:
        return copy.deepcopy(self._config["settings"])

    def get_priority_config(self) -> dict:
        """Return a flat dict keyed by category with priority + volume_multiplier."""
        cats = self._config["sound_categories"]
        return {k: {"priority": v["priority"], "volume_multiplier": v["volume_multiplier"]}
                for k, v in cats.items()}

    def set_priority(self, category: str, priority: int):
        if category not in self._config["sound_categories"]:
            raise ValueError(f"Unknown category: {category}")
        self._config["sound_categories"][category]["priority"] = priority
        self._save()

    def set_volume_multiplier(self, category: str, multiplier: float):
        if category not in self._config["sound_categories"]:
            raise ValueError(f"Unknown category: {category}")
        if not 0.0 <= multiplier <= 1.5:
            raise ValueError("multiplier must be in [0.0, 1.5]")
        self._config["sound_categories"][category]["volume_multiplier"] = multiplier
        self._save()

    def swap_priorities(self, cat_a: str, cat_b: str):
        cats = self._config["sound_categories"]
        pa = cats[cat_a]["priority"]
        pb = cats[cat_b]["priority"]
        cats[cat_a]["priority"] = pb
        cats[cat_b]["priority"] = pa
        self._save()

    def update_setting(self, key: str, value):
        self._config["settings"][key] = value
        self._save()

    def reset_to_defaults(self):
        self._config = self._load_defaults()
        self._save()

    def sorted_categories(self) -> list[tuple[str, dict]]:
        """Return categories sorted by priority (ascending = highest first)."""
        cats = self._config["sound_categories"]
        return sorted(cats.items(), key=lambda kv: kv[1]["priority"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info(f"Loaded user config from {self._path}")
                return
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupt user config ({e}), using defaults")

        self._config = self._load_defaults()
        self._save()

    @staticmethod
    def _load_defaults() -> dict:
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)
        logger.debug(f"Config saved to {self._path}")

    def __repr__(self) -> str:
        lines = ["PriorityManager:"]
        for cat, info in self.sorted_categories():
            lines.append(
                f"  [{info['priority']}] {info['display_name']}: "
                f"vol×{info['volume_multiplier']:.2f}"
            )
        return "\n".join(lines)
