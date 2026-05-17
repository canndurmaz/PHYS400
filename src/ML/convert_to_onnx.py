"""One-shot converter: Keras checkpoints → ONNX.

Walks ``alloy_model*.keras`` in the script directory and emits matching
``.onnx`` files. The ONNX bundle replaces TensorFlow in the packaged
desktop app: a 750-parameter MLP has no business shipping ~500 MB of TF
runtime when the same forward pass runs out of a ~30 MB ``onnxruntime``.

Numerical equivalence is verified per checkpoint with a fixed random
batch (so re-running the converter is reproducible and any silent
divergence between TF and ONNX surfaces immediately).
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import tensorflow as tf
import tf2onnx

from nn_alloy import ALL_ELEMENTS, C_SCALE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Opset 15 is conservative enough for any onnxruntime ≥ 1.10 and supports
# every layer this network uses (Dense, ReLU). Bumping it just bloats the
# proto without benefit for a 3-layer MLP.
_OPSET = 15
# Verification tolerance: ONNX uses FP32 just like the Keras model, so the
# only sources of disagreement are op ordering and possible kernel choice.
# 1e-5 max-abs is generous and catches anything resembling a real bug.
_ATOL = 1e-5


def convert_one(keras_path: str) -> str:
    """Convert a single .keras file to a sibling .onnx; return its path."""
    out_path = keras_path[:-len(".keras")] + ".onnx"
    print(f"  {os.path.basename(keras_path)}  →  {os.path.basename(out_path)}")
    model = tf.keras.models.load_model(keras_path)

    # Keras 3 saves models without the Keras-2 style output naming that
    # ``tf2onnx.convert.from_keras`` expects, which manifests as a
    # ``KeyError: 'keras_tensor_N'`` mid-conversion. Wrapping the forward
    # pass in a ``tf.function`` whose output is given an explicit name and
    # converting via ``from_function`` bypasses the lookup entirely.
    spec = tf.TensorSpec((None, len(ALL_ELEMENTS)), tf.float32, name="composition")

    @tf.function(input_signature=[spec])
    def serving(composition):
        return {"cij_scaled": model(composition, training=False)}

    tf2onnx.convert.from_function(
        serving, input_signature=[spec], opset=_OPSET, output_path=out_path,
    )

    # ── Verify: random batch, TF vs ORT, max-abs diff ───────────────────
    import onnxruntime as ort
    rng = np.random.RandomState(0)
    x = rng.rand(8, len(ALL_ELEMENTS)).astype(np.float32)
    tf_out = model(x, training=False).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    diff = float(np.max(np.abs(tf_out - onnx_out)))
    status = "OK" if diff <= _ATOL else "MISMATCH"
    print(f"     max|TF − ONNX| = {diff:.2e}   [{status}]")
    if status == "MISMATCH":
        raise SystemExit(f"ONNX output diverges from Keras for {keras_path}")
    return out_path


def main() -> int:
    candidates = sorted(glob.glob(os.path.join(_THIS_DIR, "alloy_model*.keras")))
    if not candidates:
        print("No alloy_model*.keras found in", _THIS_DIR)
        return 1
    print(f"Found {len(candidates)} checkpoint(s); opset={_OPSET}, C_scale={C_SCALE}")
    for p in candidates:
        convert_one(p)
    print(f"\nDone. ONNX models written next to their .keras sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
