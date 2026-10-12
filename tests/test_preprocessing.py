"""
Unit tests for dataset label maps and loaders in data/preprocessing.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import ESC50_MAP, CATEGORIES, SAMPLE_RATE, load_folder

# The 50 class names used in ESC-50's meta/esc50.csv
ESC50_CLASSES = {
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep",
    "crow", "rain", "sea_waves", "crackling_fire", "crickets",
    "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush",
    "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing",
    "coughing", "footsteps", "laughing", "brushing_teeth", "snoring",
    "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner",
    "clock_alarm", "clock_tick", "glass_breaking", "helicopter", "chainsaw",
    "siren", "car_horn", "engine", "train", "church_bells", "airplane",
    "fireworks", "hand_saw",
}


def test_esc50_map_uses_real_class_names():
    assert set(ESC50_MAP) == ESC50_CLASSES
    assert set(ESC50_MAP.values()) <= set(CATEGORIES)


def test_esc50_sirens_and_horns_are_alarms():
    assert ESC50_MAP["siren"] == "alarm"
    assert ESC50_MAP["car_horn"] == "alarm"


def test_load_folder_reads_category_subfolders(tmp_path):
    import soundfile as sf
    (tmp_path / "speech").mkdir()
    (tmp_path / "unknown").mkdir()
    tone = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    sf.write(tmp_path / "speech" / "a.wav", tone, SAMPLE_RATE)
    sf.write(tmp_path / "unknown" / "b.wav", tone, SAMPLE_RATE)
    (tmp_path / "speech" / "notes.txt").write_text("not audio")

    items = list(load_folder(str(tmp_path)))
    assert [cat for _, cat in items] == ["speech"]
    assert len(items[0][0]) == SAMPLE_RATE   # padded to CLIP_DURATION
