import numpy as np

from common.geometry import normalized_distance
from config import TALK_CYCLES, TALK_MAX_GAP


def is_ok(points):
    # Thumb and index touching, remaining fingers extended.
    pinch = normalized_distance(points, 4, 8) < 0.33
    extended = sum(
        np.linalg.norm(points[tip] - points[0]) >
        1.12 * np.linalg.norm(points[pip] - points[0])
        for tip, pip in ((12, 10), (16, 14), (20, 18))
    )
    return bool(pinch and extended >= 2)


class TalkDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = None
        self.cycles = 0
        self.last_change = None
        self.last_seen = None

    def update(self, aperture, now):
        self.last_seen = now
        # Hysteresis avoids noise around the threshold.
        if aperture >= 1.00:
            phase = 'OPEN'
        elif aperture <= 0.68:
            phase = 'CLOSED'
        else:
            return False
        if self.last_change is not None and now - self.last_change > TALK_MAX_GAP:
            self.phase, self.cycles, self.last_change = None, 0, None
        if self.phase is None:
            self.phase, self.last_change = phase, now
            return False
        if phase == self.phase:
            return False
        # Require a minimum duration between state transitions.
        if now - self.last_change < 0.12:
            return False
        previous = self.phase
        self.phase, self.last_change = phase, now
        if previous == 'CLOSED' and phase == 'OPEN':
            self.cycles += 1
            if self.cycles >= TALK_CYCLES:
                self.reset()
                return True
        return False


def get_category(result, index):
    if index >= len(result.gestures) or not result.gestures[index]:
        return 'UNKNOWN'
    cat = result.gestures[index][0]
    return cat.category_name if cat.category_name != 'None' and cat.score >= 0.55 else 'UNKNOWN'
