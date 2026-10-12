"""
Measure how well YAMNet (1 s input) classifies a labelled set of audio files.

Each file is cut into 1 s windows; every window is classified on its own
(no temporal smoothing) and the file's prediction is the category with the
highest mean probability over its windows. A window whose YAMNet scores are
all low is predicted as "unknown" and counts as a miss.

Labels come from folder names anywhere in a file's path: one of our category
names (alarm/, speech/, music/, keyboard/, environment/) or an ESC-50 /
UrbanSound8K class name (siren/, dog_bark/, ...).

Usage:
  python training/evaluate_yamnet.py data.zip
  python training/evaluate_yamnet.py data_dir/ --output results/yamnet_eval.md
  python training/evaluate_yamnet.py data.zip --yamnet-model models/yamnet

If tfhub.dev is unreachable, build models/yamnet with training/export_yamnet.py.
"""

import argparse
import datetime
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import (
    AUDIO_EXTENSIONS, CATEGORIES, ESC50_MAP, SAMPLE_RATE, URBANSOUND_MAP,
    load_audio,
)
from src.audio_classifier import (
    YAMNET_HUB_URL, YAMNET_INPUT_SECONDS, YAMNetClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
DETECTION_THRESHOLD = 0.3          # config/default_priorities.json default
MIN_LAST_WINDOW_SECONDS = 0.5      # shorter trailing remainders are dropped

LABEL_ALIASES = {**URBANSOUND_MAP, **ESC50_MAP, **{c: c for c in CATEGORIES}}
MAX_LISTED_MISTAKES = 100


def label_for(rel_path: Path):
    """Category from the nearest folder name that we recognise, else None."""
    for part in reversed(rel_path.parent.parts):
        category = LABEL_ALIASES.get(part.lower())
        if category:
            return category
    return None


def windows(audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Split into consecutive YAMNET_INPUT_SECONDS windows."""
    n = int(sr * YAMNET_INPUT_SECONDS)
    if len(audio) <= n:
        return [audio]
    out = [audio[i:i + n] for i in range(0, len(audio) - n + 1, n)]
    rest = len(audio) % n
    if rest >= int(sr * MIN_LAST_WINDOW_SECONDS):
        out.append(audio[-rest:])
    return out


def predict(probs: dict) -> str:
    if max(probs.values()) - min(probs.values()) < 1e-9:   # flat = nothing recognised
        return UNKNOWN
    return max(probs, key=probs.get)


def _member_name(info: zipfile.ZipInfo) -> str:
    """Zip member name; names written without the UTF-8 flag (e.g. by Windows
    Explorer) are usually cp949 on Korean systems, not cp437."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def audio_files(data: Path):
    """
    Return (count, iterator of (relative path, local file)) for every audio
    file in a directory or zip. Zip members are extracted one at a time to a
    temporary file, so a large archive never has to be unpacked in full.
    """
    if data.suffix.lower() != ".zip":
        files = sorted(p for p in data.rglob("*")
                       if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
        return len(files), ((p.relative_to(data), p) for p in files)

    with zipfile.ZipFile(data) as zf:
        members = [(Path(_member_name(i)), i) for i in zf.infolist()
                   if not i.is_dir() and Path(i.filename).suffix.lower() in AUDIO_EXTENSIONS]

    def extract_each():
        with zipfile.ZipFile(data) as zf, tempfile.TemporaryDirectory() as tmp:
            for rel, info in sorted(members, key=lambda m: str(m[0])):
                local = Path(tmp) / f"clip{rel.suffix.lower()}"
                with zf.open(info) as src, open(local, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                yield rel, local

    return len(members), extract_each()


def evaluate(data: Path, clf: YAMNetClassifier) -> dict:
    total, files = audio_files(data)
    y_true, y_pred, w_true, w_pred = [], [], [], []
    skipped, detected, mistakes = [], 0, []

    for i, (rel, path) in enumerate(files, 1):
        label = label_for(rel)
        if label is None:
            skipped.append(str(rel))
            continue
        try:
            audio = load_audio(str(path), target_sr=SAMPLE_RATE)
        except Exception as e:
            logger.warning(f"Skipping {rel}: {e}")
            skipped.append(str(rel))
            continue

        window_probs = []
        for w in windows(audio):
            probs = clf.classify(w, SAMPLE_RATE)
            if probs is None:
                raise RuntimeError("YAMNet inference failed")
            window_probs.append(probs)
            w_true.append(label)
            w_pred.append(predict(probs))
            detected += max(probs.values()) >= DETECTION_THRESHOLD

        mean = {c: float(np.mean([p[c] for p in window_probs])) for c in CATEGORIES}
        pred = predict(mean)
        y_true.append(label)
        y_pred.append(pred)
        if pred != label:
            mistakes.append((str(rel), label, pred))
        if i % 50 == 0:
            logger.info(f"{i}/{total} files")

    return dict(y_true=y_true, y_pred=y_pred, w_true=w_true, w_pred=w_pred,
                skipped=skipped, detected=detected, mistakes=mistakes)


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def report(r: dict, dataset: str, model: str) -> str:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    if not r["y_true"]:
        raise SystemExit("No labelled audio files found")

    present = [c for c in CATEGORIES if c in r["y_true"]]
    file_acc = accuracy_score(r["y_true"], r["y_pred"])
    win_acc = accuracy_score(r["w_true"], r["w_pred"])
    prec, rec, f1, sup = precision_recall_fscore_support(
        r["y_true"], r["y_pred"], labels=present, zero_division=0)
    cols = present + [UNKNOWN]
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=cols)

    out = [
        "# YAMNet 분류 정확도",
        "",
        f"- 측정일: {datetime.date.today().isoformat()}",
        f"- 데이터: `{dataset}` — 파일 {len(r['y_true'])}개, 1초 구간 {len(r['w_true'])}개"
        + (f" (라벨 없음/읽기 실패로 제외 {len(r['skipped'])}개)" if r["skipped"] else ""),
        f"- 모델: YAMNet ({model}), 입력 {YAMNET_INPUT_SECONDS:g}초, 클래스 이름 기반 카테고리 매핑",
        "- 파일 예측: 1초 구간별 확률 평균의 최댓값",
        "",
        "## 요약",
        "",
        _table(["지표", "값"], [
            ["파일 단위 정확도", f"{file_acc:.1%}"],
            ["1초 구간 단위 정확도", f"{win_acc:.1%}"],
            [f"감지율 (구간 최대 확률 ≥ {DETECTION_THRESHOLD})",
             f"{r['detected'] / len(r['w_true']):.1%}"],
            ["매크로 F1", f"{np.mean(f1):.3f}"],
        ]),
        "",
        "## 카테고리별 (파일 단위)",
        "",
        _table(["카테고리", "정밀도", "재현율", "F1", "파일 수"],
               [[c, f"{p:.3f}", f"{rc:.3f}", f"{f:.3f}", int(s)]
                for c, p, rc, f, s in zip(present, prec, rec, f1, sup)]),
        "",
        "## 혼동 행렬 (행: 정답, 열: 예측)",
        "",
        _table(["정답 \\ 예측"] + cols,
               [[c] + [int(v) for v in row] for c, row in zip(present, cm)]),
    ]
    if r["mistakes"]:
        listed = r["mistakes"][:MAX_LISTED_MISTAKES]
        out += ["", f"## 틀린 파일 ({len(r['mistakes'])}개"
                + (f", 앞의 {len(listed)}개만 표시" if len(listed) < len(r["mistakes"]) else "")
                + ")", "", _table(["파일", "정답", "예측"], listed)]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate YAMNet on labelled audio")
    parser.add_argument("data", help="Zip file or directory of labelled audio")
    parser.add_argument("--yamnet-model", default=YAMNET_HUB_URL,
                        help="TF Hub URL or local path of the YAMNet SavedModel")
    parser.add_argument("--output", default="results/yamnet_eval.md",
                        help="Markdown report path")
    args = parser.parse_args()

    clf = YAMNetClassifier(smoothing_window=1, model_handle=args.yamnet_model)
    if not clf.is_loaded():
        raise SystemExit(f"Could not load YAMNet from {args.yamnet_model}")

    data = Path(args.data)
    results = evaluate(data, clf)

    model = args.yamnet_model
    if not model.startswith(("http://", "https://")):
        model = f"로컬 SavedModel {Path(model).name} — 공식 yamnet.h5 가중치"
    text = report(results, data.name, model)
    print(text)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info(f"Report written to {out}")


if __name__ == "__main__":
    main()
