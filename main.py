import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from llm_chat import LLMChat
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = Path(__file__).resolve().parent
MODEL = BASE / 'gesture_recognizer.task'
CAMERA_INDEX = 1
CAPTURE_SIZE = (640, 480)
DISPLAY_SIZE = (960, 720)
COLOURS = {'Left': (255, 210, 40), 'Right': (210, 80, 255)}  # BGR
PALM = [(0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17), (5, 9, 13), (5, 13, 17)]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
TALK_CYCLES = 3
TALK_MAX_GAP = 1.3
DONE_HOLD = 1.0
CANCEL_HOLD = 2.0
MAX_RECORD_SECONDS = 60.0
MIN_EVENT_SECONDS = 0.18


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


def normalized_distance(points, a, b):
    scale = max(float(np.linalg.norm(points[5] - points[17])), 1.0)
    return float(np.linalg.norm(points[a] - points[b]) / scale)


def mouth_aperture(points):
    # Median thumb-to-fingertip distance, normalized by palm width.
    return float(np.median([normalized_distance(points, 4, i) for i in (8, 12, 16, 20)]))


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


def save_session(events, duration, frames=None):
    folder = BASE / 'recordings'
    folder.mkdir(exist_ok=True)
    name = datetime.now().strftime('gesture_%Y%m%d_%H%M%S_%f.json')
    path = folder / name
    payload = {
        'format': 'gesture-landmarks-v2',
        'duration_seconds': round(duration, 3),
        'events': events,
        'frames': frames or [],
        'meaning': None,
        'label_status': 'awaiting_asl_fingerspelling',
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved:', path)
    return path, payload


def request_interpretation(llm, payload):
    # An unlabelled landmark trajectory has no reliable semantic meaning.
    # Only send known gesture labels; never ask an LLM to invent an interpretation.
    labels = [e['gesture'] for e in payload['events']]
    if not labels:
        return 'No recognised gesture labels; sequence saved for training.'
    prompt = (
        'These are MediaPipe gesture categories, not verified user-defined commands: '
        + ', '.join(labels)
        + '. Describe only what can be established. If their meaning is unknown, '
        'say that the user must teach the meaning using ASL fingerspelling. '
        'Do not claim to have performed an action.'
    )
    return llm.chat(prompt)


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(f'Missing model: {MODEL}')
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {CAMERA_INDEX}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    filters = {side: OneEuroFilter() for side in COLOURS}
    talk = {side: TalkDetector() for side in COLOURS}
    holds = {side: {'OK': None, 'FIST': None} for side in COLOURS}
    state = 'IDLE'
    events = []
    recorded_frames = []
    recording_start = None
    active = {side: None for side in COLOURS}
    active_start = {side: None for side in COLOURS}
    smoothing = True
    start = time.monotonic()
    previous_frame = start
    last_timestamp = -1
    fps_samples = deque(maxlen=20)
    status = 'Talk gesture x3 to start'
    await_ok_release = False
    llm = LLMChat()
    executor = ThreadPoolExecutor(max_workers=1)
    llm_future = None
    llm_result = ''

    def flush_event(side, now):
        label = active[side]
        if (state == 'RECORDING' and recording_start is not None
                and label is not None and active_start[side] is not None):
            t0 = active_start[side]
            if now - t0 >= MIN_EVENT_SECONDS:
                events.append({
                    'hand': side,
                    'gesture': label,
                    'start': round(t0 - recording_start, 3),
                    'end': round(now - recording_start, 3),
                })
        active[side], active_start[side] = None, None

    cv2.namedWindow('Gesture to LLM', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture to LLM', *DISPLAY_SIZE)
    print('START: talking hand x3 | DONE: OK hold 1s | CANCEL: fist hold 2s')
    print('S: smoothing | R: reset tracking | Q/ESC: quit')
    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now = time.monotonic()
                ts = max(int((now - start) * 1000), last_timestamp + 1)
                last_timestamp = ts
                result = recognizer.recognize_for_video(image, ts)
                frame_hands = {'Left': None, 'Right': None}
                detected = set()
                labels = {side: 'NO HAND' for side in COLOURS}
                start_trigger = False
                done_trigger = False
                cancel_trigger = False

                for i, hand in enumerate(result.hand_landmarks):
                    if i >= len(result.handedness) or not result.handedness[i]:
                        continue
                    side = result.handedness[i][0].category_name
                    if side not in COLOURS or side in detected:
                        continue
                    detected.add(side)
                    frame_hands[side] = [[round(p.x, 6), round(p.y, 6), round(p.z, 6)] for p in hand]
                    raw = np.array([[p.x * w, p.y * h] for p in hand], dtype=np.float32)
                    points = filters[side].update(raw, now)
                    draw_hand(frame, points if smoothing else raw, COLOURS[side])
                    gesture = get_category(result, i)
                    ok_sign = is_ok(points)
                    fist = gesture == 'Closed_Fist'
                    labels[side] = 'OK' if ok_sign else gesture

                    if state == 'IDLE':
                        if talk[side].update(mouth_aperture(points), now):
                            start_trigger = True
                    elif state == 'RECORDING':
                        for name, condition in (('OK', ok_sign), ('FIST', fist)):
                            if condition:
                                if holds[side][name] is None:
                                    holds[side][name] = now
                            else:
                                holds[side][name] = None
                        if holds[side]['FIST'] is not None and now - holds[side]['FIST'] >= CANCEL_HOLD:
                            cancel_trigger = True
                        if holds[side]['OK'] is not None and now - holds[side]['OK'] >= DONE_HOLD and not await_ok_release:
                            done_trigger = True
                        # Exclude reserved control gestures from the input sequence.
                        label = None if ok_sign or fist or gesture == 'UNKNOWN' else gesture
                        if label != active[side]:
                            flush_event(side, now)
                            if label is not None:
                                active[side], active_start[side] = label, now

                # A fresh DONE requires releasing the previous OK sign first.
                if await_ok_release and not any(holds[s]['OK'] is not None for s in COLOURS):
                    await_ok_release = False

                for side in COLOURS:
                    if side not in detected:
                        filters[side].reset()
                        talk[side].reset()
                        holds[side] = {'OK': None, 'FIST': None}
                        if state == 'RECORDING':
                            flush_event(side, now)

                if state == 'IDLE' and start_trigger:
                    state = 'RECORDING'
                    recording_start = now
                    events = []
                    recorded_frames = []
                    for side in COLOURS:
                        talk[side].reset()
                        holds[side] = {'OK': None, 'FIST': None}
                        active[side], active_start[side] = None, None
                    await_ok_release = False
                    status = 'RECORDING - OK 1s to finish / FIST 2s to cancel'
                    print('START: recording')
                elif state == 'RECORDING':
                    recorded_frames.append({'t': round(now - recording_start, 4), 'hands': frame_hands})
                    if cancel_trigger:
                        state = 'IDLE'
                        events = []
                        recorded_frames = []
                        recording_start = None
                        status = 'CANCELLED - talk x3 to start again'
                        for side in COLOURS:
                            talk[side].reset()
                            holds[side] = {'OK': None, 'FIST': None}
                            active[side], active_start[side] = None, None
                        await_ok_release = False
                        print('CANCEL: discarded')
                    elif done_trigger or now - recording_start >= MAX_RECORD_SECONDS:
                        for side in COLOURS:
                            flush_event(side, now)
                        duration = now - recording_start
                        finished_by_done = done_trigger
                        saved_path, payload = save_session(events, duration, recorded_frames)
                        state = 'AWAIT_ASL_LABEL' if finished_by_done else 'IDLE'
                        recording_start = None
                        events = []
                        recorded_frames = []
                        for side in COLOURS:
                            talk[side].reset()
                            holds[side] = {'OK': None, 'FIST': None}
                            active[side], active_start[side] = None, None
                        if finished_by_done:
                            status = 'What did that mean? ASL A-Z recogniser not yet installed'
                            print('DONE: What did that mean?')
                            print('Raw two-hand landmarks saved:', saved_path)
                            print('ASL fingerspelling requires a separate A-Z model; no label inferred.')
                            print('Press N to start a new recording, or Q to quit.')
                        else:
                            status = '60s timeout - recording saved; talk x3 to restart'
                        # Do not call the LLM before an ASL meaning has been learned.
                        # Raw coordinates and MediaPipe categories are not semantic commands.

                if llm_future is not None and llm_future.done():
                    try:
                        llm_result = llm_future.result()
                        print('LLM:', llm_result)
                    except Exception as exc:
                        llm_result = f'LLM error: {exc}'
                        print(llm_result)
                    llm_future = None
                dt = now - previous_frame
                previous_frame = now
                if dt > 0:
                    fps_samples.append(1 / dt)
                fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0
                cv2.rectangle(frame, (0, 0), (w, 155), (18, 18, 18), -1)
                cv2.putText(frame, f'STATE: {state}', (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                            (70, 245, 110) if state == 'RECORDING' else (245, 245, 245), 2)
                cv2.putText(frame, f"LEFT: {labels['Left']}", (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOURS['Left'], 2)
                cv2.putText(frame, f"RIGHT: {labels['Right']}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOURS['Right'], 2)
                cv2.putText(frame, f'FPS: {fps:.1f} | Events: {len(events)} | Smooth: {smoothing}',
                            (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1)
                cv2.putText(frame, status[:78], (12, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 230, 255), 1)
                cv2.imshow('Gesture to LLM', cv2.resize(frame, DISPLAY_SIZE))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('n') and state == 'AWAIT_ASL_LABEL':
                    state = 'IDLE'
                    status = 'Talk gesture x3 to start (previous sequence remains unlabelled)'
                if key == ord('s'):
                    smoothing = not smoothing
                if key == ord('r'):
                    for f in filters.values():
                        f.reset()
                    for detector in talk.values():
                        detector.reset()
                    print('Tracking filters reset')
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
