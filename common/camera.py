"""Camera setup with verification against actual captured frames."""

import time


def open_camera(cv, index, size):
    cap = cv.VideoCapture(index, cv.CAP_AVFOUNDATION)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f'Cannot open camera index {index}. Check CAMERA_INDEX, macOS camera '
                'permission, and whether the selected camera is connected and available.'
            )
        cap.set(cv.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, size[1])
        actual_size = None
        # Allow initial frames from the previous capture format to drain.
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                actual_size = (frame.shape[1], frame.shape[0])
                if actual_size == size:
                    break
            else:
                time.sleep(0.02)
        if actual_size is None:
            raise RuntimeError(f'Camera {index} opened but returned no frames')
        print(f'Camera {index}: requested {size[0]}x{size[1]}, '
              f'actual {actual_size[0]}x{actual_size[1]}')
        if actual_size != size:
            print('WARNING: requested resolution was not delivered. '
                  'Check CAMERA_INDEX and CAPTURE_SIZE in config.py.')
        return cap
    except BaseException:
        cap.release()
        raise


def read_camera_frame(cv, cap, timeout=5.0):
    """Keep the window responsive during capture failures and stop on disconnect."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        if cv.waitKey(1) & 0xFF in (ord('q'), 27):
            return None
        time.sleep(0.02)
    raise RuntimeError('Camera returned no frames for 5 seconds. '
                       'Check its connection, camera permission, and other camera apps.')


def preview():
    """Run a raw-camera preview without MediaPipe or landmark drawing."""
    import argparse
    import cv2
    from config import CAMERA_INDEX, CAPTURE_SIZE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, default=CAMERA_INDEX)
    parser.add_argument('--width', type=int, default=CAPTURE_SIZE[0])
    parser.add_argument('--height', type=int, default=CAPTURE_SIZE[1])
    args = parser.parse_args()
    cap = open_camera(cv2, args.index, (args.width, args.height))
    try:
        cv2.namedWindow('Camera preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera preview', 960, 540)
        first = True
        while True:
            frame = read_camera_frame(cv2, cap)
            if frame is None:
                break
            if first:
                print(f'Raw frame: {frame.shape[1]}x{frame.shape[0]}, '
                      f'mean pixel value: {frame.mean():.1f}/255 (Q/ESC to quit)')
                first = False
            cv2.imshow('Camera preview', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    preview()
