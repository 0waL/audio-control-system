"""
Tests for training/evaluate_yamnet.py with a fake classifier (no YAMNet needed).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.evaluate_yamnet import UNKNOWN, evaluate, label_for, report, windows

SR = 16000


def test_label_from_category_or_dataset_folder():
    assert label_for(Path("set/speech/a.wav")) == "speech"
    assert label_for(Path("esc/siren/x/b.wav")) == "alarm"
    assert label_for(Path("street_music/c.wav")) == "music"
    assert label_for(Path("misc/d.wav")) is None


def test_windows_are_one_second():
    assert [len(w) for w in windows(np.zeros(SR // 2))] == [SR // 2]
    assert [len(w) for w in windows(np.zeros(int(SR * 2.6)))] == [SR, SR, int(SR * 0.6)]
    assert [len(w) for w in windows(np.zeros(int(SR * 2.2)))] == [SR, SR]


class _FakeClassifier:
    """Says 'speech' for loud audio, nothing recognised for silence."""

    def classify(self, audio, sr):
        if np.abs(audio).max() < 1e-3:
            return {c: 0.2 for c in ("alarm", "speech", "music", "keyboard", "environment")}
        return {"alarm": 0.1, "speech": 0.6, "music": 0.1, "keyboard": 0.1, "environment": 0.1}


def test_evaluate_and_report(tmp_path):
    import soundfile as sf
    for folder, loud in [("speech", True), ("music", True), ("environment", False)]:
        (tmp_path / folder).mkdir()
        audio = np.random.randn(int(SR * 1.5)).astype(np.float32) * (0.1 if loud else 0.0)
        sf.write(tmp_path / folder / "clip.wav", audio, SR)
    (tmp_path / "unlabelled").mkdir()
    sf.write(tmp_path / "unlabelled" / "x.wav", np.zeros(SR, np.float32), SR)

    r = evaluate(tmp_path, _FakeClassifier())
    assert sorted(zip(r["y_true"], r["y_pred"])) == [
        ("environment", UNKNOWN), ("music", "speech"), ("speech", "speech")]
    assert len(r["w_true"]) == 6          # 1.5 s files → 1 s + 0.5 s windows
    assert r["skipped"] == ["unlabelled/x.wav"]

    text = report(r, "test", "fake")
    assert "| 파일 단위 정확도 | 33.3% |" in text


def test_zip_is_read_member_by_member(tmp_path):
    import zipfile
    import soundfile as sf
    sf.write(tmp_path / "a.wav", np.random.randn(SR).astype(np.float32) * 0.1, SR)
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(tmp_path / "a.wav", "set/speech/a.wav")
        zf.write(tmp_path / "a.wav", "set/music/b.wav")
        zf.writestr("set/readme.txt", "not audio")

    r = evaluate(archive, _FakeClassifier())
    assert sorted(r["y_true"]) == ["music", "speech"]
    assert r["skipped"] == []


def test_korean_zip_names_decoded():
    import zipfile
    from training.evaluate_yamnet import _member_name
    info = zipfile.ZipInfo("말소리/a.wav".encode("cp949").decode("cp437"))
    assert _member_name(info) == "말소리/a.wav"
