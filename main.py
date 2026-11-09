import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from common.camera import open_camera, read_camera_frame
from common.drawing import draw_hand
from common.filters import OneEuroFilter
from common.geometry import mouth_aperture
from common.recording import SessionRecorder
from config import (
    CAMERA_INDEX, CANCEL_HOLD, CAPTURE_SIZE, COLOURS, DISPLAY_FULLSCREEN, DISPLAY_SIZE,
    DONE_HOLD, MAX_RECORD_SECONDS, MIN_EVENT_SECONDS, MODEL, RECORDINGS_DIR,
)
from gestures.detectors import TalkDetector, get_category, is_ok
from models.llm_chat import LLMChat


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(f'Missing model: {MODEL}')
    cap = open_camera(cv2, CAMERA_INDEX, CAPTURE_SIZE)
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
    recorder = None
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

    fullscreen = DISPLAY_FULLSCREEN
    print('START: talking hand x3 | DONE: OK hold 1s | CANCEL: fist hold 2s')
    print('F: fullscreen | S: smoothing | R: reset tracking | Q/ESC: quit')
    try:
        cv2.namedWindow('Gesture to LLM', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gesture to LLM', *DISPLAY_SIZE)
        # Render and process window events before initializing the model or fullscreen.
        initial_frame = read_camera_frame(cv2, cap)
        if initial_frame is None:
            return
        cv2.imshow('Gesture to LLM', cv2.flip(initial_frame, 1))
        cv2.waitKey(1)
        if fullscreen:
            cv2.setWindowProperty('Gesture to LLM', cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            while True:
                frame = read_camera_frame(cv2, cap)
                if frame is None:
                    break
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
                    recorder = SessionRecorder(RECORDINGS_DIR)
                    for side in COLOURS:
                        talk[side].reset()
                        holds[side] = {'OK': None, 'FIST': None}
                        active[side], active_start[side] = None, None
                    await_ok_release = False
                    status = 'RECORDING - OK 1s to finish / FIST 2s to cancel'
                    print('START: recording')
                elif state == 'RECORDING':
                    recorder.append_frame({'t': round(now - recording_start, 4), 'hands': frame_hands})
                    if cancel_trigger:
                        state = 'IDLE'
                        events = []
                        recorder.cancel()
                        recorder = None
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
                        saved_path = recorder.finish(events, duration)
                        recorder = None
                        state = 'AWAIT_ASL_LABEL' if finished_by_done else 'IDLE'
                        recording_start = None
                        events = []
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
                cv2.putText(frame, f'{w}x{h} | FPS: {fps:.1f} | Events: {len(events)} | Smooth: {smoothing}',
                            (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1)
                cv2.putText(frame, status[:78], (12, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 230, 255), 1)
                # Pass original pixels to HighGUI; preserve aspect ratio as the window scales.
                cv2.imshow('Gesture to LLM', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('f'):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty('Gesture to LLM', cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:
                        cv2.resizeWindow('Gesture to LLM', *DISPLAY_SIZE)
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
        if recorder is not None:
            recorder.close()
        executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
