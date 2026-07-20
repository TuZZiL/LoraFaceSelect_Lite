from pathlib import Path

import numpy as np

import lora_face_select_lite.analysis as analysis
from lora_face_select_lite.models import FaceObservation
from lora_face_select_lite.body import BodyObservation
from lora_face_select_lite.nudenet import NudeNetDetection, NudeNetResult


class StaticBackend:
    def __init__(self, faces: list[FaceObservation]) -> None:
        self.faces = faces

    def analyze(self, image: np.ndarray) -> list[FaceObservation]:
        return self.faces


def _sharp_image() -> np.ndarray:
    grid = np.where(np.indices((160, 160)).sum(axis=0) % 2, 224, 32).astype(np.uint8)
    return np.repeat(grid[:, :, None], 3, axis=2)


def _face(embedding: list[float], bbox: tuple[float, float, float, float] = (30, 30, 130, 130)) -> FaceObservation:
    return FaceObservation(
        embedding=np.asarray(embedding, dtype=np.float32),
        bbox=bbox,
        detection_score=0.95,
        yaw=0,
        pitch=30,
    )


def test_analysis_reports_progress_and_ignores_pitch_bins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: _sharp_image())
    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    progress: list[tuple[int, int, Path]] = []

    records = analysis.analyze_dataset(
        paths,
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([1.0, 0.0])]),
        min_similarity=0.35,
        progress=lambda current, total, path: progress.append((current, total, path)),
    )

    assert [record.status for record in records] == ["eligible", "eligible"]
    assert all(record.pose_bin == "frontal" for record in records)
    assert progress == [(1, 2, paths[0]), (2, 2, paths[1])]


def test_analysis_rejects_catastrophically_bad_quality(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: np.zeros((160, 160, 3), dtype=np.uint8))

    record = analysis.analyze_dataset(
        [tmp_path / "black.jpg"],
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([1.0, 0.0])]),
        min_similarity=0.35,
    )[0]

    assert record.status == "rejected"
    assert record.reason == "low_quality_blur"
    assert not record.fallback_eligible


def test_expensive_backends_are_skipped_for_hard_rejections(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: _sharp_image())

    class UnexpectedBackend:
        def embed(self, *args):
            raise AssertionError("appearance backend must not run")

        def analyze(self, *args, **kwargs):
            raise AssertionError("body backend must not run")

    record = analysis.analyze_dataset(
        [tmp_path / "wrong_person.jpg"],
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([0.0, 1.0])]),
        min_similarity=0.50,
        appearance_identity=(np.ones(2), np.ones(2)),
        appearance_backend=UnexpectedBackend(),
        body_backend=UnexpectedBackend(),
    )[0]

    assert record.reason == "below_similarity"
    assert record.appearance_similarity is None
    assert record.body_shape is None


def test_multiple_faces_select_best_match_when_faces_do_not_overlap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: _sharp_image())
    path = tmp_path / "group.jpg"

    accepted = analysis.analyze_dataset(
        [path],
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([1.0, 0.0], (10, 30, 90, 130)), _face([0.0, 1.0], (130, 35, 159, 75))]),
        min_similarity=0.35,
    )[0]
    rejected = analysis.analyze_dataset(
        [path],
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([0.0, 1.0]), _face([0.0, -1.0])]),
        min_similarity=0.35,
    )[0]

    assert (accepted.status, accepted.reason) == ("eligible", "")
    assert accepted.face_count == 2
    assert accepted.metadata["target_bbox"] == [10.0, 30.0, 90.0, 130.0]
    assert accepted.metadata["other_face_bboxes"] == [[130.0, 35.0, 159.0, 75.0]]
    assert (rejected.status, rejected.reason) == ("rejected", "below_similarity")


def test_overlapping_target_and_other_face_stays_in_review(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: _sharp_image())

    record = analysis.analyze_dataset(
        [tmp_path / "overlap.jpg"],
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([1.0, 0.0]), _face([0.0, 1.0], (100, 40, 150, 100))]),
        min_similarity=0.35,
    )[0]

    assert (record.status, record.reason) == ("multiple_faces_review", "faces_overlap")


def test_close_head_zones_stay_in_review_even_when_face_boxes_do_not_overlap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: _sharp_image())

    record = analysis.analyze_dataset(
        [tmp_path / "close_heads.jpg"],
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([1.0, 0.0]), _face([0.0, 1.0], (140, 40, 159, 100))]),
        min_similarity=0.35,
    )[0]

    assert (record.status, record.reason) == ("multiple_faces_review", "faces_overlap")


def test_analysis_rejects_extreme_pose_for_training(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: _sharp_image())
    face = _face([1.0, 0.0])
    face.yaw = 31.0

    record = analysis.analyze_dataset(
        [tmp_path / "difficult_pose.jpg"],
        np.asarray([1.0, 0.0]),
        StaticBackend([face]),
        min_similarity=0.35,
        max_abs_yaw=25.0,
    )[0]

    assert (record.status, record.reason) == ("rejected", "pose_too_extreme")


def test_analysis_runs_nudenet_only_on_matched_target_body_roi(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis, "read_image", lambda path, max_side: _sharp_image())
    landmarks = np.zeros((33, 5), dtype=np.float32)
    for index, point in {11: (45, 50), 12: (115, 50), 23: (50, 125), 24: (110, 125)}.items():
        landmarks[index] = (*point, 0, 1, 1)
    body = BodyObservation((35, 35, 125, 150), landmarks, None, 0.9, 0.95)

    class BodyBackend:
        def analyze(self, image, face_bbox):
            assert face_bbox == (30, 30, 130, 130)
            return body

    class NudeNetBackend:
        def __init__(self):
            self.rois = []

        def detect(self, image, roi=None):
            self.rois.append(roi)
            return NudeNetResult(
                [NudeNetDetection("belly-covered", 8, 0.8, (50, 70, 30, 40))],
                provider="test",
            )

    nudenet = NudeNetBackend()
    record = analysis.analyze_dataset(
        [tmp_path / "person.jpg"],
        np.asarray([1.0, 0.0]),
        StaticBackend([_face([1.0, 0.0])]),
        min_similarity=0.35,
        body_backend=BodyBackend(),
        nudenet_backend=nudenet,
    )[0]

    assert nudenet.rois == [body.bbox]
    assert record.nudenet_labels == "belly-covered"
    assert record.nudenet_max_score == 0.8
    assert record.nudenet_error is None
