"""
Build a local YAMNet SavedModel with the same interface as
https://tfhub.dev/google/yamnet/1, for machines that cannot reach tfhub.dev.

It downloads the official weights (storage.googleapis.com/audioset/yamnet.h5)
and the YAMNet model code from tensorflow/models, then saves a SavedModel that
`tensorflow_hub.load()` — and so YAMNetClassifier — can open from disk.

Usage:
  python training/export_yamnet.py models/yamnet
  python training/evaluate_yamnet.py data.zip --yamnet-model models/yamnet
"""

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
# Pinned tensorflow/models revision of research/audioset/yamnet
CODE_URL = ("https://raw.githubusercontent.com/tensorflow/models/"
            "34a21326906b9574fa11c4d6d0a5c534ff039267/research/audioset/yamnet/")
CODE_FILES = ["yamnet.py", "params.py", "features.py", "yamnet_class_map.csv"]


def export(out_dir: Path):
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as src:
        for name in CODE_FILES:
            urllib.request.urlretrieve(CODE_URL + name, os.path.join(src, name))
        urllib.request.urlretrieve(WEIGHTS_URL, os.path.join(src, "yamnet.h5"))

        sys.path.insert(0, src)
        import params as yamnet_params
        import yamnet

        class YAMNet(tf.Module):
            def __init__(self):
                super().__init__()
                self._yamnet = yamnet.yamnet_frames_model(yamnet_params.Params())
                self._yamnet.load_weights(os.path.join(src, "yamnet.h5"))
                self._class_map_asset = tf.saved_model.Asset(
                    os.path.join(src, "yamnet_class_map.csv"))

            @tf.function(input_signature=[])
            def class_map_path(self):
                return self._class_map_asset.asset_path

            # Returns (scores, embeddings, log_mel_spectrogram) like the TF Hub model
            @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
            def __call__(self, waveform):
                return self._yamnet(waveform)

        tf.saved_model.save(YAMNet(), str(out_dir))


def main():
    parser = argparse.ArgumentParser(description="Export YAMNet to a local SavedModel")
    parser.add_argument("out_dir", nargs="?", default="models/yamnet")
    args = parser.parse_args()
    out = Path(args.out_dir)
    export(out)
    print(f"YAMNet SavedModel written to {out}")


if __name__ == "__main__":
    main()
