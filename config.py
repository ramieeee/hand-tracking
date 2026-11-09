from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL = BASE / 'models' / 'gesture_recognizer.task'
RECORDINGS_DIR = BASE / 'recordings'
CAMERA_INDEX = 1
# Built-in FaceTime camera. For a connected iPhone, check its index first;
# the previously enumerated iPhone supported (1920, 1440).
CAPTURE_SIZE = (1920, 1080)
DISPLAY_SIZE = (1280, 960)  # Initial window size when fullscreen is disabled.
DISPLAY_FULLSCREEN = False  # Toggle with F after the preview appears.
COLOURS = {'Left': (255, 210, 40), 'Right': (210, 80, 255)}  # BGR
PALM = [(0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17), (5, 9, 13), (5, 13, 17)]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
TALK_CYCLES = 3
TALK_MAX_GAP = 1.3
DONE_HOLD = 1.0
CANCEL_HOLD = 2.0
MAX_RECORD_SECONDS = 60.0
MIN_EVENT_SECONDS = 0.18
# Deliberate fingerspelling hold. Fluent coarticulated spelling is outside this baseline.
ASL_MIN_HOLD = 0.32
ASL_LETTER_GAP = 0.18

