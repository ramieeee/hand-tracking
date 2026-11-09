# Hand Tracking project

MediaPipe 손 제스처 인식 및 랜드마크 녹화 프로젝트입니다.

```text
main.py                 카메라 및 녹화 상태 흐름, 실행 진입점
config.py               경로, 카메라, 화면, 제스처 임계값 설정
models/
  gesture_recognizer.task
  hand_landmarker.task
  llm_chat.py           Ollama 클라이언트
  interpretation.py     인식된 제스처의 LLM 프롬프트
common/
  filters.py            좌표 평활화 필터
  geometry.py           거리 및 손가락 간격 계산
  drawing.py            손 랜드마크 화면 표시
  recording.py          JSONL 스트리밍 저장 및 JSON/JSONL 읽기
gestures/
  detectors.py          OK, 말하는 손, MediaPipe 카테고리 판정
recordings/             로컬 녹화 데이터
tests/                  녹화 저장 검증
```

`opencv-python`, `mediapipe`, `numpy`가 설치된 Python 환경에서 실행합니다.

```sh
python main.py
```

카메라 번호는 `config.py`의 `CAMERA_INDEX`에서 변경합니다. 현재 카메라 백엔드는 macOS의 AVFoundation을 사용합니다. LLM 클라이언트는 기본적으로 로컬 Ollama의 `qwen3.5:2b`를 사용하지만, 현재 녹화 흐름에서는 ASL 의미 라벨을 학습하기 전까지 LLM을 호출하지 않습니다.

- 시작: 말하는 손 제스처 3회
- 완료: OK 1초 유지
- 취소: 주먹 2초 유지
- `N`: 완료 후 새 녹화 대기, `S`: 평활화 전환, `R`: 추적 초기화, `Q`/ESC: 종료

## 녹화 형식

새 녹화는 세션마다 하나의 JSONL 파일로 저장합니다. 각 줄은 독립적인 JSON 객체입니다.

1. `session`: 형식 버전(`gesture-landmarks-v3`), 의미 및 라벨 상태
2. `frame`: 프레임별 시간 `t`와 양손의 정규화된 `[x, y, z]` 랜드마크
3. `event`: 완료 시 기록하는 인식 제스처 구간
4. `summary`: 완료 시 기록하는 전체 녹화 시간

프레임마다 디스크에 기록하므로 녹화 길이에 비례해 프레임 목록을 메모리에 쌓지 않습니다. JSONL은 한 줄씩 읽을 수 있고, 들여쓰기를 제거해 기존 JSON보다 저장량도 줄어듭니다. 압축 형식은 아니므로 데이터가 더 커지면 완료된 파일의 gzip 압축이나 학습용 Parquet 변환을 별도로 고려할 수 있습니다.

녹화 중에는 `.jsonl.partial` 확장자를 사용하며 완료하면 `.jsonl`로 변경합니다. 취소 시 해당 임시 파일을 삭제합니다. 녹화 도중 종료하거나 오류가 나면 미완료 파일을 남기며, 학습 데이터로 사용할 때는 완료된 `.json`/`.jsonl`만 선택해야 합니다. 프레임은 매번 flush하지만 정전까지 견디는 디스크 동기화를 보장하지는 않습니다.

기존 JSON 샘플은 그대로 보존합니다. `common.recording.load_session(path)`는 기존 JSON과 새 JSONL을 동일한 payload 구조로 읽습니다. 대용량 JSONL 처리 시에는 전체 파일을 메모리에 적재하는 `load_session` 대신 `iter_session_records(path)`로 순회합니다.

새 녹화 파일은 `.gitignore`로 Git에서 제외합니다. 이미 추적 중인 기존 샘플에는 적용되지 않습니다. 실제 녹화 데이터는 별도 스토리지나 백업으로 관리하세요.

## 검증

```sh
python -m unittest discover -s tests
```
