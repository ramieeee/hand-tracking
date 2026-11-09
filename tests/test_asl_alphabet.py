import math
import unittest

import numpy as np

from gestures.asl_alphabet import classify_asl_letter, decode_fingerspelling, hand_features


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.array([0.0, -1.0, 0.0])
    return vector / norm


def _bend_direction(direction, amount):
    """Rotate an extended direction toward the palm without cancelling to zero."""
    direction = _unit(direction)
    down = np.array([0.0, 1.0, 0.0])
    cosine = float(np.clip(np.dot(direction, down), -1.0, 1.0))
    angle = amount * math.acos(cosine)
    perpendicular = np.array([-direction[1], direction[0], 0.0])
    if float(np.dot(perpendicular, down)) < 0.0:
        perpendicular = -perpendicular
    perpendicular = _unit(perpendicular)
    return _unit(math.cos(angle) * direction + math.sin(angle) * perpendicular)


def _joints(mcp, direction, bend=0.0):
    points = []
    position = np.asarray(mcp, dtype=float)
    for step, length in enumerate((0.36, 0.28, 0.24)):
        current = _bend_direction(direction, min(1.0, bend * (step + 1) / 2.4))
        position = position + current * length
        points.append(position.copy())
    return points


def make_hand(poses, thumb='closed', orient=0.0, spread=0.0, cross=False):
    """Build one canonical 21-point hand. Distances are in palm-width units."""
    mcps = {
        'index': np.array([-0.50, -0.95, 0.0]),
        'middle': np.array([-0.16, -1.02, 0.0]),
        'ring': np.array([0.16, -0.98, 0.0]),
        'pinky': np.array([0.50, -0.90, 0.0]),
    }
    directions = {
        'index': np.array([-spread, -1.0, 0.0]),
        'middle': np.array([0.0, -1.0, 0.0]),
        'ring': np.array([spread * 0.8, -1.0, 0.0]),
        'pinky': np.array([spread, -1.0, 0.0]),
    }
    joints = {}
    for name, mcp in mcps.items():
        pose = poses.get(name, 'curled')
        if pose == 'extended':
            joints[name] = _joints(mcp, directions[name], bend=0.0)
        elif pose == 'partial':
            joints[name] = _joints(mcp, directions[name], bend=0.55)
        elif pose == 'hook':
            pip = mcp + np.array([0.02, -0.40, 0.0])
            dip = pip + np.array([0.24, -0.06, 0.0])
            tip = mcp + np.array([0.45, -0.53, 0.0])
            joints[name] = [pip, dip, tip]
        else:
            joints[name] = _joints(mcp, np.array([0.0, -1.0, 0.0]), bend=1.0)
    if cross:
        joints['index'][2] = mcps['middle'] + np.array([0.10, -0.86, 0.0])
        joints['middle'][2] = mcps['index'] + np.array([-0.10, -0.86, 0.0])

    index_tip = joints['index'][2]
    thumbs = {
        'closed': [(-0.18, -0.28), (-0.34, -0.46), (-0.46, -0.58), (-0.62, -0.70)],
        'out': [(-0.22, -0.30), (-0.55, -0.36), (-0.86, -0.34), (-1.12, -0.30)],
        'across': [(-0.16, -0.30), (-0.30, -0.48), (-0.36, -0.62), (-0.40, -0.78)],
        'tuck': [(-0.18, -0.28), (-0.32, -0.50), (-0.36, -0.82), (-0.38, -1.20)],
        'under_n': [(-0.12, -0.26), (-0.10, -0.40), (0.02, -0.58), (0.02, -0.82)],
        'under_m': [(-0.10, -0.24), (0.05, -0.38), (0.22, -0.52), (0.36, -0.72)],
        'parallel': [(-0.28, -0.32), (-0.40, -0.58), (-0.42, -0.88), (-0.40, -1.22)],
        'pinch': None,
        'fork': [(-0.20, -0.34), (-0.34, -0.58), (-0.30, -0.92), (-0.24, -1.28)],
        'e': None,
    }
    if thumb == 'pinch':
        thumb_points = [
            np.array([-0.20, -0.30, 0.0]),
            np.array([-0.34, -0.52, 0.0]),
            np.array([-0.30, -0.78, 0.0]),
            index_tip + np.array([0.06, 0.04, 0.0]),
        ]
    elif thumb == 'e':
        thumb_points = [
            np.array([-0.18, -0.30, 0.0]),
            np.array([-0.30, -0.55, 0.0]),
            np.array([-0.28, -0.78, 0.0]),
            index_tip + np.array([0.22, 0.20, 0.0]),
        ]
    else:
        thumb_points = [np.array([x, y, 0.0]) for x, y in thumbs[thumb]]
    ordered = [np.array([0.0, 0.0, 0.0]), *thumb_points]
    for name in ('index', 'middle', 'ring', 'pinky'):
        ordered.extend([mcps[name], *joints[name]])
    cosine, sine = math.cos(orient), math.sin(orient)
    rotated = []
    for point in ordered:
        rotated.append([
            cosine * point[0] - sine * point[1],
            sine * point[0] + cosine * point[1],
            point[2],
        ])
    return rotated


def _hold(landmarks, start, duration, step=0.05, shift=None):
    frames = []
    count = int(round(duration / step))
    for index in range(count + 1):
        progress = index / max(count, 1)
        offset = shift(progress) if shift else (0.0, 0.0)
        moved = [
            [point[0] + offset[0], point[1] + offset[1], point[2]]
            for point in landmarks
        ]
        frames.append({
            't': round(start + index * step, 4),
            'hands': {'Left': None, 'Right': moved},
        })
    return frames


class AlphabetShapeTests(unittest.TestCase):
    SHAPES = (
        ('A', dict(thumb='closed')),
        ('B', dict(poses={'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'},
                   thumb='across', spread=-0.12)),
        ('C', dict(poses={'index': 'partial', 'middle': 'partial', 'ring': 'partial', 'pinky': 'partial'},
                   thumb='parallel')),
        ('D', dict(poses={'index': 'extended'}, thumb='fork')),
        ('E', dict(poses={'index': 'partial', 'middle': 'partial', 'ring': 'partial', 'pinky': 'partial'},
                   thumb='e')),
        ('F', dict(poses={'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'}, thumb='pinch')),
        ('G', dict(poses={'index': 'extended'}, thumb='parallel', orient=math.pi / 2)),
        ('H', dict(poses={'index': 'extended', 'middle': 'extended'}, thumb='parallel',
                   orient=math.pi / 2, spread=-0.2)),
        ('I', dict(poses={'pinky': 'extended'}, thumb='closed')),
        ('K', dict(poses={'index': 'extended', 'middle': 'extended'}, thumb='fork', spread=0.35)),
        ('L', dict(poses={'index': 'extended'}, thumb='out')),
        ('M', dict(thumb='under_m')),
        ('N', dict(thumb='under_n')),
        ('O', dict(poses={'index': 'partial', 'middle': 'partial', 'ring': 'partial', 'pinky': 'partial'},
                   thumb='pinch')),
        ('P', dict(poses={'index': 'extended', 'middle': 'extended'}, thumb='fork', spread=0.35,
                   orient=math.pi)),
        ('Q', dict(poses={'index': 'extended'}, thumb='parallel', orient=math.pi)),
        ('R', dict(poses={'index': 'extended', 'middle': 'extended'}, thumb='closed', cross=True)),
        ('S', dict(thumb='across')),
        ('T', dict(thumb='tuck')),
        ('U', dict(poses={'index': 'extended', 'middle': 'extended'}, thumb='closed', spread=-0.18)),
        ('V', dict(poses={'index': 'extended', 'middle': 'extended'}, thumb='closed', spread=0.45)),
        ('W', dict(poses={'index': 'extended', 'middle': 'extended', 'ring': 'extended'},
                   thumb='closed', spread=0.15)),
        ('X', dict(poses={'index': 'hook'}, thumb='closed')),
        ('Y', dict(poses={'pinky': 'extended'}, thumb='out')),
    )

    def test_canonical_letters(self):
        failed = []
        for letter, options in self.SHAPES:
            poses = options.get('poses', {})
            landmarks = make_hand(
                poses,
                thumb=options.get('thumb', 'closed'),
                orient=options.get('orient', 0.0),
                spread=options.get('spread', 0.0),
                cross=options.get('cross', False),
            )
            got = classify_asl_letter(landmarks)
            if got != letter:
                features = hand_features(landmarks)
                if features is None:
                    failed.append(f'{letter}->None features')
                    continue
                states = {name: value['state'] for name, value in features['fingers'].items()}
                failed.append(
                    f'{letter}->{got} orient={features["orient"]} reach={features["thumb_reach"]:.2f} '
                    f'pinch={features["thumb_index_tip"]:.2f} across={features["thumb_across"]:.2f} '
                    f'distal={features["thumb_distal"]:.2f} cluster={features["cluster"]:.2f} '
                    f'gap={features["gaps"]["index_middle"]:.2f} '
                    f'index_mcp={features["fingers"]["index"]["mcp_across"]:.2f} '
                    f'middle_mcp={features["fingers"]["middle"]["mcp_across"]:.2f} '
                    f'ring_mcp={features["fingers"]["ring"]["mcp_across"]:.2f} '
                    f'pip={features["fingers"]["index"]["pip_distal"]:.2f} states={states}'
                )
        self.assertFalse(failed, '\n'.join(failed))

    def test_translation_and_scale_do_not_change_a_letter(self):
        landmarks = make_hand({'index': 'extended'}, thumb='out')
        moved = [
            [point[0] * 0.31 + 0.4, point[1] * 0.31 + 0.2, point[2] * 0.31]
            for point in landmarks
        ]
        self.assertEqual(classify_asl_letter(moved), 'L')

    def test_open_spread_hand_is_not_a_letter(self):
        landmarks = make_hand(
            {'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'},
            thumb='out',
            spread=0.85,
        )
        self.assertIsNone(classify_asl_letter(landmarks))


class SequenceTests(unittest.TestCase):
    def test_two_held_letters_and_done_gesture_are_separated(self):
        frames = []
        frames.extend(_hold(make_hand({'index': 'extended'}, thumb='out'), 0.0, 0.5))
        frames.extend(_empty(0.55, 0.25))
        frames.extend(_hold(
            make_hand({'index': 'extended', 'middle': 'extended'}, thumb='closed', spread=0.45),
            0.85, 0.5,
        ))
        frames.extend(_hold(
            make_hand({'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'}, thumb='pinch'),
            1.40, 1.0,
        ))
        result = decode_fingerspelling(frames, trim_done_hold=1.0)
        self.assertEqual(result['text'], 'LV')
        self.assertEqual(result['method'], 'geometric-baseline-v1')
        self.assertEqual(result['hand'], 'Right')

    def test_j_and_z_use_the_fingertip_path(self):
        jay = make_hand({'pinky': 'extended'}, thumb='closed')
        zed = make_hand({'index': 'extended'}, thumb='fork')

        def hook(progress):
            return (0.20 * math.sin(math.pi * progress), 0.60 * progress)

        def zigzag(progress):
            if progress < 0.34:
                return (0.50 * progress / 0.34, 0.0)
            if progress < 0.67:
                local = (progress - 0.34) / 0.33
                return (0.50 - 0.45 * local, 0.28 * local)
            local = (progress - 0.67) / 0.33
            return (0.05 + 0.50 * local, 0.28)

        frames = _hold(jay, 0.0, 0.7, shift=hook)
        frames.extend(_empty(0.75, 0.25))
        frames.extend(_hold(zed, 1.05, 0.7, shift=zigzag))
        self.assertEqual(decode_fingerspelling(frames)['text'], 'JZ')

    def test_bounce_repeats_a_letter(self):
        letter = make_hand({'index': 'extended'}, thumb='out')

        def bounce(progress):
            return (0.0, 0.70 * math.sin(math.pi * progress))

        result = decode_fingerspelling(_hold(letter, 0.0, 0.8, shift=bounce))
        self.assertEqual(result['text'], 'LL')

    def test_left_hand_is_used_when_it_is_the_only_hand(self):
        landmarks = make_hand({'pinky': 'extended'}, thumb='out')
        frames = [{
            't': round(index * 0.05, 4),
            'hands': {'Left': landmarks, 'Right': None},
        } for index in range(12)]
        result = decode_fingerspelling(frames)
        self.assertEqual(result['text'], 'Y')
        self.assertEqual(result['hand'], 'Left')


def _empty(start, duration, step=0.05):
    count = int(round(duration / step))
    return [{
        't': round(start + index * step, 4),
        'hands': {'Left': None, 'Right': None},
    } for index in range(count + 1)]


class EmptyGapTests(unittest.TestCase):
    def test_gap_of_missing_hands_splits_letters(self):
        first = _hold(make_hand({'index': 'extended'}, thumb='out'), 0.0, 0.45)
        second = _hold(make_hand({'pinky': 'extended'}, thumb='closed'), 0.8, 0.45)
        frames = first + _empty(0.5, 0.25) + second
        self.assertEqual(decode_fingerspelling(frames)['text'], 'LI')


if __name__ == '__main__':
    unittest.main()
