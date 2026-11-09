import json
import tempfile
import unittest
from pathlib import Path

from common.recording import SessionRecorder, iter_session_records, load_session


class RecordingTests(unittest.TestCase):
    def test_stream_and_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = SessionRecorder(folder)
            frame = {'t': 0.1, 'hands': {'Left': [[0.1, 0.2, 0.3]], 'Right': None}}
            events = [{'hand': 'Left', 'gesture': 'Open_Palm', 'start': 0, 'end': 0.5}]
            recorder.append_frame(frame)
            # The current frame is on disk before the recording finishes.
            records = list(iter_session_records(recorder.partial_path))
            self.assertEqual(records[-1], {'type': 'frame', **frame})
            self.assertFalse(recorder.path.exists())
            path = recorder.finish(events, 0.56789)
            payload = load_session(path)
            self.assertEqual(payload['frames'], [frame])
            self.assertEqual(payload['events'], events)
            self.assertEqual(payload['duration_seconds'], 0.568)
            self.assertIsNone(payload['meaning'])
            self.assertFalse(recorder.partial_path.exists())

    def test_fingerspelling_summary_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = SessionRecorder(folder)
            fingerspelling = {
                'text': 'CAT',
                'hand': 'Right',
                'letters': [],
                'method': 'geometric-baseline-v1',
            }
            path = recorder.finish([], 1.2, fingerspelling=fingerspelling)
            self.assertEqual(load_session(path)['fingerspelling'], fingerspelling)

    def test_cancel_removes_partial_data(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = SessionRecorder(folder)
            recorder.append_frame({'t': 0, 'hands': {}})
            recorder.cancel()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_interruption_keeps_partial_and_loader_rejects_it(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = SessionRecorder(folder)
            recorder.close()
            self.assertTrue(recorder.partial_path.exists())
            with self.assertRaises(ValueError):
                load_session(recorder.partial_path)

    def test_legacy_json_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.json'
            payload = {'duration_seconds': 1.0, 'events': []}
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(load_session(path), payload)

    def test_incomplete_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'incomplete.jsonl'
            path.write_text('{"type":"session","format":"gesture-landmarks-v3"}\n')
            with self.assertRaises(ValueError):
                load_session(path)


if __name__ == '__main__':
    unittest.main()
