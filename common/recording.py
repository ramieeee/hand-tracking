"""Stream recordings to JSONL and read both JSONL and legacy JSON sessions."""

import json
from datetime import datetime
from pathlib import Path


class SessionRecorder:
    def __init__(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime('gesture_%Y%m%d_%H%M%S_%f.jsonl')
        self.path = folder / name
        self.partial_path = self.path.with_suffix('.jsonl.partial')
        self._file = self.partial_path.open('x', encoding='utf-8')
        self._write({
            'type': 'session',
            'format': 'gesture-landmarks-v3',
            'meaning': None,
            'label_status': 'awaiting_asl_fingerspelling',
        })

    def _write(self, record):
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
        self._file.flush()

    def append_frame(self, frame):
        self._write({'type': 'frame', **frame})

    def finish(self, events, duration):
        for event in events:
            self._write({'type': 'event', **event})
        self._write({'type': 'summary', 'duration_seconds': round(duration, 3)})
        self.close()
        self.partial_path.rename(self.path)
        print('Saved:', self.path)
        return self.path

    def close(self):
        """Keep unfinished data as .partial so it cannot be mistaken for a complete session."""
        self._file.close()

    def cancel(self):
        self.close()
        self.partial_path.unlink(missing_ok=True)


def iter_session_records(path):
    """Read complete JSONL sessions line by line without loading every frame."""
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_session(path):
    """Load a complete session into the legacy payload shape; uses memory for all frames."""
    path = Path(path)
    if path.suffix == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    if path.suffix != '.jsonl':
        raise ValueError(f'Expected a completed .json or .jsonl recording: {path}')
    payload = {'events': [], 'frames': []}
    has_header = has_summary = False
    for record in iter_session_records(path):
        kind = record.pop('type')
        if kind == 'session':
            payload.update(record)
            has_header = True
        elif kind == 'frame':
            payload['frames'].append(record)
        elif kind == 'event':
            payload['events'].append(record)
        elif kind == 'summary':
            payload.update(record)
            has_summary = True
        else:
            raise ValueError(f'Unknown recording record type: {kind}')
    if not has_header or not has_summary:
        raise ValueError(f'Incomplete recording: {path}')
    return payload
