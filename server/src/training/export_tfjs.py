"""
Convert trained Keras models to TensorFlow.js format for the browser client, and
measure the accuracy cost of quantisation (Chapters 3.6 / 4.5).

Requires `tensorflowjs`:  pip install tensorflowjs

Run:
  python -m src.training.export_tfjs \
      --tiredness models_out/tiredness/tiredness_keras.keras \
      --boredom   models_out/boredom/boredom_keras.keras \
      --out client/models

Note: the boredom model has an image input (MobileNetV2). Running it fully in the
browser means also running MobileNetV2 in TF.js, which is heavier. The tiredness
LSTM is light and is the primary client-side model; the boredom model can either
run client-side (heavier) or be evaluated server-side. Both are exported here so
you can benchmark each path (Objective 3).
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys


def convert(keras_path: str, out_dir: str, quantize: bool):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=keras",
    ]
    if quantize:
        # 1-byte affine quantisation of weights -> ~4x smaller
        cmd.append("--quantize_uint8=*")
    cmd += [keras_path, out_dir]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiredness")
    ap.add_argument("--boredom")
    ap.add_argument("--out", default="client/models")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    quant = not args.no_quantize
    if args.tiredness:
        convert(args.tiredness, os.path.join(args.out, "tiredness"), quant)
    if args.boredom:
        convert(args.boredom, os.path.join(args.out, "boredom"), quant)
    print("Done. Point the browser client at", args.out)
    print("Reminder: compare pre/post-quantisation accuracy in your report —")
    print("re-run evaluate on the converted model to quantify the trade-off.")


if __name__ == "__main__":
    main()
