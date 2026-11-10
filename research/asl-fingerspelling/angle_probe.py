"""Synthetic sensitivity probe, NOT an ASL accuracy benchmark.

Run from the repository root:
    uv run --with numpy python research/asl-fingerspelling/angle_probe.py

Uses the existing flat, hand-built unit-test fixtures. Rotates them in 3D,
then lets the current classifier use only XY (orthographic projection).
No camera, landmark estimator, occlusion, or real signer is simulated.
CSV goes to stdout; this script does not alter recordings or runtime code.
"""

import csv
import math
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gestures.asl_alphabet import classify_asl_letter  # noqa: E402


def rotation(axis, degrees):
    theta = math.radians(degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    if axis == "yaw":
        return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    if axis == "pitch":
        return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def main():
    fixtures = runpy.run_path(str(ROOT / "tests/test_asl_alphabet.py"))
    writer = csv.writer(sys.stdout)
    writer.writerow(["experiment", "axis", "degrees", "letter", "prediction"])
    for letter, options in fixtures["AlphabetShapeTests"].SHAPES:
        points = np.asarray(fixtures["make_hand"](
            options.get("poses", {}),
            **{key: value for key, value in options.items() if key != "poses"},
        ))
        for axis in ("roll", "yaw", "pitch"):
            for degrees in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
                transformed = points @ rotation(axis, degrees).T
                prediction = classify_asl_letter(transformed)
                writer.writerow(["orthographic_rotation", axis, degrees, letter, prediction or "NONE"])
        # Fixtures are isotropic. Simulate separate image-width/height
        # normalization for square versus 16:9 images, at equal pixel scale.
        for aspect in (1.0, 16 / 9):
            normalized = points.copy()
            normalized[:, 0] /= aspect
            prediction = classify_asl_letter(normalized)
            writer.writerow(["image_aspect", str(aspect), 0, letter, prediction or "NONE"])


if __name__ == "__main__":
    main()
