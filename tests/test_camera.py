import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.camera import open_camera, read_camera_frame


class CameraTests(unittest.TestCase):
    def make_backend(self, frames):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.side_effect = frames
        cv = SimpleNamespace(VideoCapture=Mock(return_value=cap), CAP_AVFOUNDATION=1,
                             CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3)
        return cv, cap

    def test_waits_for_requested_resolution(self):
        cv, cap = self.make_backend([
            (True, SimpleNamespace(shape=(480, 640, 3))),
            (True, SimpleNamespace(shape=(1440, 1920, 3))),
        ])
        self.assertIs(open_camera(cv, 1, (1920, 1440)), cap)
        self.assertEqual(cap.read.call_count, 2)
        cap.release.assert_not_called()

    @patch('builtins.print')
    def test_warns_when_camera_ignores_resolution(self, output):
        cv, cap = self.make_backend([(True, SimpleNamespace(shape=(720, 1280, 3)))] * 10)
        self.assertIs(open_camera(cv, 1, (1920, 1440)), cap)
        self.assertTrue(any('WARNING' in call.args[0] for call in output.call_args_list))

    @patch('common.camera.time.sleep')
    def test_releases_camera_when_no_frames_arrive(self, sleep):
        cv, cap = self.make_backend([(False, None)] * 10)
        with self.assertRaisesRegex(RuntimeError, 'no frames'):
            open_camera(cv, 1, (1920, 1440))
        cap.release.assert_called_once()

    def test_releases_camera_when_open_fails(self):
        cv, cap = self.make_backend([])
        cap.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'Cannot open camera'):
            open_camera(cv, 1, (1920, 1440))
        cap.release.assert_called_once()

    @patch('common.camera.time.sleep')
    def test_read_recovers_and_pumps_window_events(self, sleep):
        frame = object()
        cv, cap = self.make_backend([(False, None), (True, frame)])
        cv.waitKey = Mock(return_value=-1)
        self.assertIs(read_camera_frame(cv, cap), frame)
        cv.waitKey.assert_called_once_with(1)

    def test_can_quit_while_camera_is_not_delivering_frames(self):
        cv, cap = self.make_backend([(False, None)])
        cv.waitKey = Mock(return_value=ord('q'))
        self.assertIsNone(read_camera_frame(cv, cap))

    @patch('common.camera.time.monotonic', side_effect=[0, 6])
    def test_disconnected_camera_does_not_wait_forever(self, monotonic):
        cv, cap = self.make_backend([])
        with self.assertRaisesRegex(RuntimeError, 'no frames'):
            read_camera_frame(cv, cap)


if __name__ == '__main__':
    unittest.main()
