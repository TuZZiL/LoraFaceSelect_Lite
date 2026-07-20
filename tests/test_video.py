from pathlib import Path

import cv2
import numpy as np

from lora_face_select_lite.cli import _build_parser
from lora_face_select_lite.models import FaceObservation
from lora_face_select_lite.video import extract_video_frames, video_paths


class FakeCapture:
    def __init__(self, _path: str) -> None:
        rng = np.random.default_rng(42)
        self.frames = [rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(20)]
        self.position_ms = 0.0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 1.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 20.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.position_ms
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_MSEC:
            self.position_ms = value
        return True

    def read(self):
        index = min(len(self.frames) - 1, round(self.position_ms / 1000.0))
        return True, self.frames[index].copy()

    def release(self) -> None:
        pass


class FakeBackend:
    provider = "fake-face"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, _image: np.ndarray) -> list[FaceObservation]:
        self.calls += 1
        return [FaceObservation(np.array([1.0, 0.0], dtype=np.float32), (20, 20, 105, 110), 0.99)]


def test_video_paths_are_recursive_and_filtered(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.mp4").write_bytes(b"video")
    (nested / "b.MOV").write_bytes(b"video")
    (nested / "ignore.jpg").write_bytes(b"image")

    assert video_paths(tmp_path) == [tmp_path / "a.mp4", nested / "b.MOV"]
    assert video_paths(tmp_path, recursive=False) == [tmp_path / "a.mp4"]


def test_video_extraction_is_bounded_and_reuses_checkpoint(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"local-video")
    output = tmp_path / "result"
    output.mkdir()
    backend = FakeBackend()
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    first = extract_video_frames(
        [video], output, np.array([1.0, 0.0], dtype=np.float32), backend,
        sample_fps=0.25, max_samples=5, max_candidates=3, analysis_size=128,
        min_similarity=0.5, min_face_width=20,
    )

    assert first.videos_processed == 1
    assert first.sampled_frames == 5
    assert 1 <= len(first.frames) <= 3
    assert all(frame.path.is_file() and frame.source_video == video for frame in first.frames)
    calls_after_first_run = backend.calls

    second = extract_video_frames(
        [video], output, np.array([1.0, 0.0], dtype=np.float32), backend,
        sample_fps=0.25, max_samples=5, max_candidates=3, analysis_size=128,
        min_similarity=0.5, min_face_width=20,
    )

    assert second.videos_reused == 1
    assert second.videos_processed == 0
    assert backend.calls == calls_after_first_run
    assert [frame.path for frame in second.frames] == [frame.path for frame in first.frames]
    assert (output / "video_progress.json").is_file()


def test_select_video_defaults_are_safe() -> None:
    args = _build_parser().parse_args([
        "select", "dataset", "--references", "reference.jpg", "--output", "result",
    ])

    assert args.analyze_videos is True
    assert args.video_sample_fps == 0.5
    assert args.video_max_samples == 120
    assert args.video_max_candidates == 3
