import cv2
import numpy as np

from config import FINGERS, PALM


def draw_hand(frame, points, colour):
    pts = np.rint(points).astype(np.int32)
    overlay = frame.copy()
    thickness = max(7, int(np.linalg.norm(points[5] - points[17]) * 0.18))
    for a, b, c in PALM:
        cv2.fillConvexPoly(overlay, pts[[a, b, c]], colour)
    for finger in FINGERS:
        for a, b in zip(finger[:-1], finger[1:]):
            cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), colour, thickness, cv2.LINE_AA)
        for idx in finger:
            cv2.circle(overlay, tuple(pts[idx]), thickness // 2, colour, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
    # Draw coloured lines directly; no black-background edge layer.
    edge = tuple(int(c * 0.65 + 255 * 0.35) for c in colour)
    drawn = set()
    for a, b, c in PALM:
        for u, v in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((u, v)))
            if key not in drawn:
                drawn.add(key)
                cv2.line(frame, tuple(pts[u]), tuple(pts[v]), edge, 1, cv2.LINE_AA)
    for finger in FINGERS:
        for a, b in zip(finger[:-1], finger[1:]):
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), edge, 1, cv2.LINE_AA)
