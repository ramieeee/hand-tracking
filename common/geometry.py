import numpy as np


def normalized_distance(points, a, b):
    scale = max(float(np.linalg.norm(points[5] - points[17])), 1.0)
    return float(np.linalg.norm(points[a] - points[b]) / scale)


def mouth_aperture(points):
    # Median thumb-to-fingertip distance, normalized by palm width.
    return float(np.median([normalized_distance(points, 4, i) for i in (8, 12, 16, 20)]))
