"""Evaluate the project's YAMNet category mapping on an UrbanSound8K ZIP.

Reads UrbanSound8K.csv and audio directly from the archive, so the dataset
does not need to be extracted. Predictions are saved after each file and a
later invocation resumes from the saved CSV.
"""

import argparse
import csv
import io
import logging
import math
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import CATEGORIES, SAMPLE_RATE, URBANSOUND_MAP
from src.audio_classifier import YAMNET_HUB_URL, YAMNetClassifier
from training.evaluate_yamnet import DETECTION_THRESHOLD, UNKNOWN, predict, windows

logger = logging.getLogger(__name__)
FIELDS = ["fold", "filename", "source_class", "label", "prediction",
          "windows", "correct_windows", "detected_windows", "window_predictions", "error"]


def load_wav_from_zip(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with sf.SoundFile(io.BytesIO(archive.read(member))) as audio_file:
        audio = audio_file.read(dtype="float32", always_2d=True).mean(axis=1)
        source_rate = audio_file.samplerate
    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(source_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor,
                              source_rate // divisor).astype(np.float32)
    return audio


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def write_report(records, archive_name, output):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    valid = [r for r in records if not r["error"]]
    errors = [r for r in records if r["error"]]
    if not valid:
        raise ValueError("No audio files were evaluated successfully")
    true = [r["label"] for r in valid]
    pred = [r["prediction"] for r in valid]
    labels = CATEGORIES + [UNKNOWN]
    correct = sum(a == b for a, b in zip(true, pred))
    total_windows = sum(int(r["windows"]) for r in valid)
    correct_windows = sum(int(r["correct_windows"]) for r in valid)
    detected_windows = sum(int(r["detected_windows"]) for r in valid)
    matrix = confusion_matrix(true, pred, labels=labels)
    by_source = []
    for source in sorted({r["source_class"] for r in valid}):
        group = [r for r in valid if r["source_class"] == source]
        hits = sum(r["label"] == r["prediction"] for r in group)
        by_source.append([source, URBANSOUND_MAP[source], hits, len(group),
                          f"{hits / len(group):.1%}"])

    lines = [
        "# YAMNet on UrbanSound8K",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Dataset: `{archive_name}`; {len(valid)} evaluated files, {len(errors)} read/inference errors",
        "- Model: TF Hub google/yamnet/1 with this project's fixed 521-to-5 class mapping; no trained head",
        "- Audio: mono, 16 kHz; non-overlapping 1-second windows; final remainder included if at least 0.5 seconds",
        "- File prediction: highest mean category probability over its windows; no temporal smoothing",
        "- Ground truth: UrbanSound8K.csv mapped through `URBANSOUND_MAP`",
        "- UrbanSound8K contains no speech or keyboard labels, so those categories are not measured for recall",
        "",
        "## Overall",
        "",
        _table(["Metric", "Result"], [
            ["File accuracy", f"{correct}/{len(valid)} = {accuracy_score(true, pred):.2%}"],
            ["Window accuracy", f"{correct_windows}/{total_windows} = {correct_windows / total_windows:.2%}"],
            [f"Window detection (maximum category probability >= {DETECTION_THRESHOLD})",
             f"{detected_windows}/{total_windows} = {detected_windows / total_windows:.2%}"],
            ["Macro F1 (represented labels only)",
             f"{f1_score(true, pred, labels=sorted(set(true)), average='macro', zero_division=0):.3f}"],
        ]),
        "",
        "## Accuracy by UrbanSound8K class",
        "",
        _table(["Source class", "Mapped category", "Correct", "Files", "Accuracy"], by_source),
        "",
        "## Confusion matrix (rows: true, columns: predicted)",
        "",
        _table(["True / predicted"] + labels,
               [[label] + [int(x) for x in row] for label, row in zip(labels, matrix)]),
    ]
    if errors:
        lines += ["", "## Files not scored", "",
                  _table(["File", "Error"], [[r["filename"], r["error"]] for r in errors])]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report written to %s", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--limit", type=int, help="Score only the first N files")
    parser.add_argument("--folds", help="Comma-separated UrbanSound8K folds to score")
    parser.add_argument("--predictions", type=Path, default=Path("results/urbansound8k_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/yamnet_urbansound8k.md"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with zipfile.ZipFile(args.archive) as archive:
        metadata = list(csv.DictReader(io.TextIOWrapper(
            archive.open("UrbanSound8K.csv"), encoding="utf-8-sig")))
        if args.folds:
            selected_folds = set(args.folds.split(","))
            metadata = [row for row in metadata if row["fold"] in selected_folds]
        if args.limit is not None:
            metadata = metadata[:args.limit]
        members = set(archive.namelist())
        missing = [row for row in metadata if f"fold{row['fold']}/{row['slice_file_name']}" not in members]
        if missing:
            raise ValueError(f"{len(missing)} metadata rows have no matching WAV")

        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        done = {}
        if args.predictions.exists():
            with args.predictions.open(newline="", encoding="utf-8") as source:
                done = {row["filename"]: row for row in csv.DictReader(source)}
        clf = YAMNetClassifier(smoothing_window=1, model_handle=YAMNET_HUB_URL)
        if not clf.is_loaded():
            raise RuntimeError("YAMNet could not be loaded")
        with args.predictions.open("a", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            if target.tell() == 0:
                writer.writeheader()
            for index, row in enumerate(metadata, 1):
                filename = f"fold{row['fold']}/{row['slice_file_name']}"
                if filename in done:
                    continue
                label = URBANSOUND_MAP[row["class"]]
                record = dict(fold=row["fold"], filename=filename,
                              source_class=row["class"], label=label,
                              prediction="", windows=0, correct_windows=0,
                              detected_windows=0, window_predictions="", error="")
                try:
                    audio = load_wav_from_zip(archive, filename)
                    predictions = []
                    probabilities = []
                    for window in windows(audio):
                        probs = clf.classify(window, SAMPLE_RATE)
                        if probs is None:
                            raise RuntimeError("YAMNet inference failed")
                        probabilities.append(probs)
                        predictions.append(predict(probs))
                    means = {category: float(np.mean([p[category] for p in probabilities]))
                             for category in CATEGORIES}
                    record.update(prediction=predict(means), windows=len(predictions),
                                  correct_windows=sum(p == label for p in predictions),
                                  detected_windows=sum(max(p.values()) >= DETECTION_THRESHOLD
                                                       for p in probabilities),
                                  window_predictions=";".join(predictions))
                except Exception as error:
                    record["error"] = str(error)
                    logger.warning("Failed %s: %s", filename, error)
                writer.writerow(record)
                target.flush()
                done[filename] = record
                if index % 100 == 0:
                    logger.info("Processed %s/%s files", index, len(metadata))

    selected = {f"fold{row['fold']}/{row['slice_file_name']}" for row in metadata}
    write_report([done[name] for name in sorted(selected)], args.archive.name, args.output)


if __name__ == "__main__":
    main()
