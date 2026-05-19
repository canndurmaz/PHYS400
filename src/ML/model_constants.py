"""TF-free constants and algebra used by both training and inference.

Split out of ``nn_alloy.py`` so that the runtime predictor (and the
Flask UI it backs) can import these without pulling TensorFlow into the
process. The packaged desktop app ships ``onnxruntime`` only, so any
transitive TF import would defeat the ~10× bundle-size reduction we get
from the ONNX conversion.

``nn_alloy.py`` re-exports these for backwards compatibility with any
caller that has been importing them from there historically.
"""

from __future__ import annotations

import numpy as np

# Composition basis. Order matters: the trained model's input column ``i``
# corresponds to the mole fraction of ``ALL_ELEMENTS[i]``. Adding an
# element here means retraining; reordering means rewriting checkpoints.
ALL_ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Mo", "Ni", "Si", "Ti", "Zn"]

# Output normalisation: most C_ij values land in 50–400 GPa, so we divide
# by 200 before training and multiply back at inference. Lives here (not
# only in nn_alloy) because the ONNX model emits the *scaled* tensor and
# the predictor needs to undo it without importing the trainer.
C_SCALE = 200.0


def cij_to_e_nu(C11, C12):
    """Cubic-isotropic algebra: (C11, C12) → (E, ν). Vectorised, safe at
    small sums by clamping the denominator's sign to avoid divide-by-zero.
    """
    C11 = np.asarray(C11)
    C12 = np.asarray(C12)
    s = C11 + C12
    s_safe = np.where(np.abs(s) < 1e-8, np.sign(s) * 1e-8 + 1e-12, s)
    E = (C11 - C12) * (C11 + 2.0 * C12) / s_safe
    nu = C12 / s_safe
    return E, nu
