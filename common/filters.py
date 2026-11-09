import math

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff=0.8, beta=0.025):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.reset()

    def reset(self):
        self.prev = self.prev_raw = self.prev_dx = self.prev_time = None

    @staticmethod
    def alpha(dt, cutoff):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def update(self, points, now):
        if self.prev is None:
            self.prev, self.prev_raw = points.copy(), points.copy()
            self.prev_dx = np.zeros_like(points)
            self.prev_time = now
            return points.copy()
        dt = max(0.001, min(now - self.prev_time, 0.1))
        dx = (points - self.prev_raw) / dt
        a_dx = self.alpha(dt, 1.0)
        dx_hat = a_dx * dx + (1 - a_dx) * self.prev_dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.alpha(dt, cutoff)
        filtered = a * points + (1 - a) * self.prev
        self.prev, self.prev_raw, self.prev_dx, self.prev_time = (
            filtered.copy(), points.copy(), dx_hat.copy(), now
        )
        return filtered
