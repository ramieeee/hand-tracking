"""Geometric baseline for the ASL manual alphabet.

Sources and the limits of this baseline are recorded in research/asl-fingerspelling/.

MediaPipe already supplies 21 landmarks per hand. This module turns one
hand's landmarks into a letter, then groups held letters into a fingerspelled
string. J and Z are the two letters that require a path, not only a pose.
The rules are a baseline specification, not a trained signer-independent model.
"""

from collections import deque

import numpy as np

from config import ASL_LETTER_GAP, ASL_MIN_HOLD

METHOD = 'geometric-baseline-v1'

_FINGERS = (
    ('index', 5, 6, 7, 8),
    ('middle', 9, 10, 11, 12),
    ('ring', 13, 14, 15, 16),
    ('pinky', 17, 18, 19, 20),
)


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros(vector.shape, dtype=float)
    return vector / norm


def hand_features(landmarks):
    """Palm-width-normalised finger state for one 21-point hand, or None."""
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    palm = float(np.linalg.norm(points[5, :2] - points[17, :2]))
    if palm < 1e-4:
        return None
    local = (points - points[0]) / palm
    distal = _unit(local[9, :2])
    across = _unit(local[5, :2] - local[17, :2])
    if float(np.linalg.norm(distal)) < 0.5 or float(np.linalg.norm(across)) < 0.5:
        return None

    def project(index, axis):
        return float(np.dot(local[index, :2], axis))

    fingers = {}
    for name, mcp, pip, _dip, tip in _FINGERS:
        pip_length = float(np.linalg.norm(local[pip, :2]))
        tip_length = float(np.linalg.norm(local[tip, :2]))
        ratio = tip_length / max(pip_length, 1e-6)
        straight = float(np.dot(
            _unit(local[pip, :2] - local[mcp, :2]),
            _unit(local[tip, :2] - local[pip, :2]),
        ))
        # Hook is a bent finger whose tip has not folded back to the palm.
        if ratio >= 1.15 and straight >= 0.55:
            state = 'extended'
        elif ratio >= 0.98 and straight < 0.45:
            state = 'hook'
        elif ratio < 1.06 and straight < 0.45:
            state = 'curled'
        else:
            state = 'partial'
        fingers[name] = {
            'state': state,
            'ratio': ratio,
            'straight': straight,
            'mcp_across': project(mcp, across),
            'pip_distal': project(pip, distal),
            'tip_distal': project(tip, distal),
        }

    # Palm direction stays valid when the index finger is curled. G, H, P, and Q rotate the whole hand.
    palm_image = points[9, :2] - points[0, :2]
    absolute = abs(float(np.arctan2(palm_image[0], -palm_image[1])))
    if absolute < 0.85:
        orient = 'upright'
    elif absolute < 2.25:
        orient = 'sideways'
    else:
        orient = 'down'
    gaps = {
        'index_middle': float(np.linalg.norm(local[8, :2] - local[12, :2])),
        'middle_ring': float(np.linalg.norm(local[12, :2] - local[16, :2])),
        'ring_pinky': float(np.linalg.norm(local[16, :2] - local[20, :2])),
    }
    mcp_order = project(5, across) - project(9, across)
    tip_order = project(8, across) - project(12, across)
    return {
        'fingers': fingers,
        'orient': orient,
        'thumb_reach': float(np.linalg.norm(local[4, :2] - local[5, :2])),
        'thumb_align': float(np.dot(
            _unit(local[4, :2] - local[2, :2]),
            _unit(local[8, :2] - local[5, :2]),
        )),
        'thumb_index_tip': float(np.linalg.norm(local[4, :2] - local[8, :2])),
        'thumb_across': project(4, across),
        'thumb_distal': project(4, distal),
        'gaps': gaps,
        'crossed': bool(mcp_order * tip_order < 0 and gaps['index_middle'] < 0.55),
        'cluster': float(np.mean([
            np.linalg.norm(local[tip, :2] - local[8, :2]) for tip in (12, 16, 20)
        ])),
    }


def _states(features):
    return {name: features['fingers'][name]['state'] for name in ('index', 'middle', 'ring', 'pinky')}


def _fist_letter(features):
    """Separate the closed letters by where the thumb sits across the knuckles."""
    thumb = features['thumb_across']
    index = features['fingers']['index']['mcp_across']
    middle = features['fingers']['middle']['mcp_across']
    ring = features['fingers']['ring']['mcp_across']
    if thumb > index + 0.06:
        return 'A'
    if middle < thumb <= index + 0.06:
        pip = features['fingers']['index']['pip_distal']
        if features['thumb_distal'] > pip - 0.08:
            return 'T'
        return 'S'
    if ring < thumb <= middle:
        return 'N'
    return 'M'


def classify_asl_letter(landmarks):
    """Return one static ASL letter, or None when the pose is not a letter."""
    features = hand_features(landmarks)
    if features is None:
        return None
    state = _states(features)
    extended = {name: value == 'extended' for name, value in state.items()}
    curled = {name: value == 'curled' for name, value in state.items()}
    orient = features['orient']
    reach = features['thumb_reach']
    pinch = features['thumb_index_tip']
    gaps = features['gaps']
    thumb_out = reach >= 0.58

    if extended['pinky'] and curled['index'] and curled['middle'] and curled['ring']:
        return 'Y' if thumb_out else 'I'
    if pinch <= 0.42 and extended['middle'] and extended['ring'] and extended['pinky'] and not extended['index']:
        return 'F'
    if (extended['index'] and extended['middle'] and extended['ring'] and curled['pinky']
            and orient == 'upright' and gaps['index_middle'] >= 0.28 and gaps['middle_ring'] >= 0.22):
        return 'W'
    if all(extended.values()) and orient == 'upright':
        if max(gaps.values()) <= 0.46 and not thumb_out:
            return 'B'
        return None
    two = extended['index'] and extended['middle'] and curled['ring'] and curled['pinky']
    if two:
        if features['crossed']:
            return 'R'
        if orient == 'sideways' and gaps['index_middle'] < 0.46:
            return 'H'
        if orient == 'down' and reach >= 0.36:
            return 'P'
        if orient == 'upright' and gaps['index_middle'] < 0.30:
            return 'U'
        if orient == 'upright' and reach >= 0.36:
            return 'K'
        if orient == 'upright':
            return 'V'
        return None
    others_curled = curled['middle'] and curled['ring'] and curled['pinky']
    if others_curled and state['index'] in ('extended', 'hook'):
        if state['index'] == 'hook' and orient == 'upright':
            return 'X'
        if state['index'] == 'extended' and orient == 'sideways':
            return 'G'
        if state['index'] == 'extended' and orient == 'down':
            return 'Q'
        if state['index'] == 'extended' and thumb_out and orient == 'upright':
            return 'L'
        if state['index'] == 'extended' and orient == 'upright':
            return 'D'

    if any(extended.values()):
        return None
    partials = sum(value == 'partial' for value in state.values())
    tip_distal = np.mean([features['fingers'][name]['tip_distal'] for name in state])
    if partials >= 3 and pinch <= 0.16:
        return 'O'
    if partials >= 3 and pinch >= 0.55 and orient == 'upright':
        return 'C'
    if partials >= 2 and pinch <= 0.58 and features['thumb_distal'] + 0.05 < tip_distal:
        return 'E'
    if all(curled.values()):
        return _fist_letter(features)
    return None


def _is_hook(path):
    points = np.asarray(path, dtype=float)
    if len(points) < 4:
        return False
    chord = points[-1] - points[0]
    length = float(np.linalg.norm(chord))
    if length < 0.35:
        return False
    direction = chord / length
    normal = np.array([-direction[1], direction[0]])
    deviation = float(np.max(np.abs((points - points[0]) @ normal)))
    return deviation >= 0.12


def _is_zigzag(path):
    points = np.asarray(path, dtype=float)
    if len(points) < 6:
        return False
    axis = 0 if np.ptp(points[:, 0]) >= np.ptp(points[:, 1]) else 1
    series = points[:, axis]
    reversals = 0
    direction = 0
    stroke = 0.0
    for start, end in zip(series[:-1], series[1:]):
        delta = float(end - start)
        if abs(delta) < 1e-4:
            continue
        sign = 1 if delta > 0 else -1
        if direction == 0 or sign == direction:
            direction = sign
            stroke += abs(delta)
            continue
        if stroke >= 0.22:
            reversals += 1
        direction = sign
        stroke = abs(delta)
    return reversals >= 2


def _is_bounce(path):
    points = np.asarray(path, dtype=float)
    if len(points) < 6:
        return False
    distance = np.linalg.norm(points - points[0], axis=1)
    peak = int(np.argmax(distance))
    return bool(distance[peak] >= 0.45 and distance[-1] <= 0.22 and 0 < peak < len(points) - 1)


def _scaled_point(landmarks, index):
    points = np.asarray(landmarks, dtype=float)
    palm = max(float(np.linalg.norm(points[5, :2] - points[17, :2])), 1e-6)
    return (points[index, :2] / palm).tolist()


class FingerspellingDecoder:
    """Commit a letter after it holds, then read J, Z, and a doubled letter from its path."""

    def __init__(self, min_hold=ASL_MIN_HOLD, gap=ASL_LETTER_GAP):
        self.min_hold = min_hold
        self.gap = gap
        self._committed = []
        self._current = None
        self._window = deque(maxlen=3)

    def update(self, timestamp, hands):
        hand, letter = self._smooth(*self._observe(hands))
        landmarks = (hands or {}).get(hand) if hand else None
        if letter is None or landmarks is None:
            if self._current is not None and timestamp - self._current['last'] >= self.gap:
                self._close_current()
            return
        if self._current is None or self._current['letter'] != letter:
            self._close_current()
            self._current = {
                'letter': letter,
                'hand': hand,
                'start': timestamp,
                'end': timestamp,
                'last': timestamp,
                'wrist': [_scaled_point(landmarks, 0)],
                'index_tip': [_scaled_point(landmarks, 8)],
                'pinky_tip': [_scaled_point(landmarks, 20)],
            }
            return
        self._current['end'] = timestamp
        self._current['last'] = timestamp
        self._current['wrist'].append(_scaled_point(landmarks, 0))
        self._current['index_tip'].append(_scaled_point(landmarks, 8))
        self._current['pinky_tip'].append(_scaled_point(landmarks, 20))

    def finish(self, end_time, trim_done_hold=0.0):
        self._close_current()
        boundary = None if trim_done_hold <= 0 else end_time - trim_done_hold
        letters = []
        for item in self._committed:
            start, end = item['start'], item['end']
            if boundary is not None:
                if start >= boundary - 1e-9:
                    continue
                end = min(end, boundary)
            if end - start < self.min_hold and not item.get('repeat'):
                continue
            if end - start <= 1e-9:
                continue
            letters.append({
                'letter': item['letter'],
                'hand': item['hand'],
                'start': round(start, 3),
                'end': round(end, 3),
            })
        return {
            'text': ''.join(item['letter'] for item in letters),
            'hand': letters[0]['hand'] if letters else None,
            'letters': letters,
            'method': METHOD,
        }

    def committed_text(self):
        return ''.join(item['letter'] for item in self._committed)

    def _observe(self, hands):
        found = {}
        for side in ('Right', 'Left'):
            landmarks = (hands or {}).get(side)
            found[side] = classify_asl_letter(landmarks) if landmarks else None
        active = self._current['hand'] if self._current is not None else None
        if active is not None:
            return active, found[active]
        if found['Right']:
            return 'Right', found['Right']
        if found['Left']:
            return 'Left', found['Left']
        return None, None

    def _smooth(self, hand, letter):
        self._window.append((hand, letter))
        counts = {}
        for sample_hand, sample_letter in self._window:
            counts[sample_letter] = counts.get(sample_letter, 0) + 1
        best_letter, best_count = max(
            counts.items(),
            key=lambda item: (item[1], item[0] is not None),
        )
        if best_count < 2:
            return hand, None
        for sample_hand, sample_letter in reversed(self._window):
            if sample_letter == best_letter:
                return sample_hand, best_letter
        return hand, None

    def _close_current(self):
        segment = self._current
        self._current = None
        if segment is None or segment['end'] - segment['start'] < self.min_hold:
            return
        letter = segment['letter']
        if letter == 'I' and _is_hook(segment['pinky_tip']):
            letter = 'J'
        elif letter in ('D', 'G', 'X') and _is_zigzag(segment['index_tip']):
            letter = 'Z'
        self._committed.append({
            'letter': letter,
            'hand': segment['hand'],
            'start': segment['start'],
            'end': segment['end'],
        })
        if letter not in ('J', 'Z') and _is_bounce(segment['wrist']):
            midpoint = segment['start'] + 0.5 * (segment['end'] - segment['start'])
            self._committed.append({
                'letter': letter,
                'hand': segment['hand'],
                'start': midpoint,
                'end': segment['end'],
                'repeat': True,
            })


def decode_fingerspelling(frames, trim_done_hold=0.0, min_hold=ASL_MIN_HOLD, gap=ASL_LETTER_GAP):
    """Decode recording frames of the form {'t', 'hands': {'Left', 'Right'}}."""
    decoder = FingerspellingDecoder(min_hold, gap)
    end = 0.0
    for frame in frames:
        end = frame['t']
        decoder.update(frame['t'], frame['hands'])
    return decoder.finish(end, trim_done_hold=trim_done_hold)
