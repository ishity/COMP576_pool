"""Simulation package.

Compatibility note: pooltool 0.3.3 calls ``np.math.atan2`` (removed in numpy 2.x)
inside ``pooltool.ptmath.angle_between_vectors``. We restore the alias here before
pooltool is imported anywhere in our code.
"""
import math

import numpy as np

if not hasattr(np, "math"):
    np.math = math  # type: ignore[attr-defined]
