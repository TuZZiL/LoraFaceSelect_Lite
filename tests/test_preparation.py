from pathlib import Path

import numpy as np
from PIL import Image

from lora_face_select_lite.models import CandidateRecord
from lora_face_select_lite.preparation import CropSkip, _crop_risks, body_aware_crop, face_safe_bounds, fit_crop_to_training_bucket, parsing_region, prepare_dataset, recommended_crop


def test_recommended_crop_preserves_context_for_far_images() -> None:
    box, strategy = recommended_crop((1200, 800), (500, 200, 620, 340), "far")
    assert box == (0, 0, 1200, 800)
    assert strategy == "context_original"


def test_smart_crop_stays_inside_image_and_contains_face() -> None:
    face = (5.0, 5.0, 105.0, 125.0)
    box, strategy = recommended_crop((800, 1000), face, "close", (0, 0, 140, 180))
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 800
    assert 0 <= y1 < y2 <= 1000
    assert x1 <= face[0] and y1 <= face[1] and x2 >= face[2] and y2 >= face[3]
    assert strategy == "smart_close"


def test_parsing_region_is_bounded() -> None:
    assert parsing_region((0, 0, 100, 120), 640, 480) == (0, 0, 336, 336)


def test_crop_is_expanded_to_exact_training_bucket_aspect() -> None:
    box, bucket = fit_crop_to_training_bucket((400, 200, 800, 1400), (1600, 1800), 512, 1024)
    assert bucket == (512, 1024)
    assert (box[2] - box[0]) * bucket[1] == (box[3] - box[1]) * bucket[0]


def test_small_source_has_no_compatible_training_bucket() -> None:
    box, bucket = fit_crop_to_training_bucket((0, 0, 400, 700), (400, 700), 512, 1024)
    assert box == (0, 0, 400, 700)
    assert bucket is None


def test_whole_image_is_trimmed_to_exact_bucket_around_target() -> None:
    box, bucket = fit_crop_to_training_bucket(
        (0, 0, 1440, 1800),
        (1440, 1800),
        512,
        1024,
        required_box=(1050, 250, 1250, 500),
    )

    assert bucket == (768, 1024)
    assert (box[2] - box[0]) * bucket[1] == (box[3] - box[1]) * bucket[0]
    assert box[0] <= 1050 and box[1] <= 250 and box[2] >= 1250 and box[3] >= 500


def test_face_safe_bounds_keep_target_and_exclude_other_face() -> None:
    target = (400.0, 100.0, 600.0, 350.0)
    other = (720.0, 120.0, 900.0, 340.0)
    bounds = face_safe_bounds((1000, 1200), target, [other], (250, 20, 850, 900))

    assert bounds is not None
    assert bounds[0] <= target[0] and bounds[2] >= target[2]
    assert bounds[2] < other[0]


def test_face_safe_bounds_reject_overlapping_faces() -> None:
    assert face_safe_bounds(
        (1000, 1200),
        (400.0, 100.0, 600.0, 350.0),
        [(550.0, 150.0, 750.0, 400.0)],
        (250, 20, 850, 900),
    ) is None


def test_face_safe_bounds_reject_close_overlapping_head_zones() -> None:
    assert face_safe_bounds(
        (1000, 1200),
        (400.0, 100.0, 600.0, 350.0),
        [(630.0, 140.0, 800.0, 360.0)],
        (250, 20, 850, 900),
    ) is None


class FakeParser:
    def predict(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[10:-10, 10:-10] = 17
        return mask


class FakeBody:
    def __init__(self, size=(1000, 1000), confidence=0.99) -> None:
        self.confidence = confidence
        self.detection_score = 0.9
        self.landmarks = np.zeros((33, 5), dtype=np.float32)
        self.mask = np.zeros((size[1], size[0]), dtype=np.uint8)

    def analyze(self, image: np.ndarray, bbox: tuple[float, float, float, float]) -> "FakeBody":
        return self


def test_body_crop_falls_back_when_pose_is_uncertain() -> None:
    decision = body_aware_crop((1000, 1000), (400, 100, 600, 350), "medium", (350, 30, 650, 400), FakeBody(confidence=0.2))
    assert decision.box == (0, 0, 1000, 1000)
    assert decision.reasons == ("pose_uncertain",)


def test_body_crop_accepts_a_clear_safe_portrait() -> None:
    decision = body_aware_crop((1000, 1200), (400, 150, 600, 400), "medium", (350, 80, 650, 450), FakeBody((1000, 1200)))
    assert decision.safety == "safe"
    assert decision.box != (0, 0, 1000, 1200)


def test_crop_risks_detect_fingers_crossing_boundary() -> None:
    body = FakeBody()
    body.landmarks[15] = (450, 300, 0, 1, 1)
    body.landmarks[17] = (750, 300, 0, 1, 1)
    risks = _crop_risks((300, 100, 600, 700), (1000, 1000), (380, 120, 570, 350), body)
    assert "fingers_cut" in risks


def test_crop_risks_allow_non_hand_limb_crossing() -> None:
    body = FakeBody()
    body.landmarks[11] = (450, 300, 0, 1, 1)
    body.landmarks[13] = (750, 300, 0, 1, 1)
    risks = _crop_risks((300, 100, 600, 700), (1000, 1000), (380, 120, 570, 350), body)
    assert risks == ()


def test_wrist_uses_forearm_length_as_finger_safety_margin() -> None:
    body = FakeBody()
    body.landmarks[13] = (300, 300, 0, 1, 1)
    body.landmarks[15] = (500, 300, 0, 1, 1)
    risks = _crop_risks((100, 100, 550, 700), (1000, 1000), (200, 120, 450, 350), body)
    assert "hand_near_edge" in risks


def test_existing_source_edge_is_not_reported_as_a_new_head_cut() -> None:
    body = FakeBody()
    risks = _crop_risks((200, 0, 800, 800), (1000, 1000), (300, 0, 700, 350), body)
    assert "head_near_edge" not in risks


def test_prepare_dataset_writes_crops_manifest_and_review(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (1600, 2000), (100, 120, 140)).save(source)
    record = CandidateRecord(
        source, "eligible", similarity=0.8, scale_bin="medium", pose_bin="frontal",
        metadata={"target_bbox": [700, 300, 900, 550]},
    )

    rows = prepare_dataset(tmp_path / "result", [record], FakeParser(), max_side=768, body_backend=FakeBody((1600, 2000)))

    assert len(rows) == 1
    assert Path(str(rows[0]["prepared"])).is_file()
    assert rows[0]["resolution_status"] == "ready"
    assert rows[0]["training_bucket"] in {"512x768", "768x1024"}
    assert rows[0]["was_upscaled"] is False
    assert (tmp_path / "result" / "dataset_manifest.csv").is_file()
    assert (tmp_path / "result" / "parsing_preview" / "001_source.jpg").is_file()
    review = (tmp_path / "result" / "review.html").read_text(encoding="utf-8")
    assert "Рекомендований кроп" in review


def test_prepare_dataset_preserves_video_provenance(tmp_path: Path) -> None:
    source = tmp_path / "frame.jpg"
    video = tmp_path / "clip.mp4"
    Image.new("RGB", (1600, 2000), (100, 120, 140)).save(source)
    record = CandidateRecord(
        source, "eligible", similarity=0.8, scale_bin="medium", pose_bin="frontal",
        metadata={
            "target_bbox": [700, 300, 900, 550],
            "source_video": str(video),
            "video_timestamp_seconds": 12.5,
            "video_frame_number": 375,
        },
    )

    rows = prepare_dataset(tmp_path / "result", [record], FakeParser(), max_side=768, body_backend=FakeBody((1600, 2000)))

    assert rows[0]["source_video"] == str(video)
    assert rows[0]["video_timestamp_seconds"] == 12.5
    assert rows[0]["video_frame_number"] == 375


def test_prepare_dataset_excludes_other_detected_faces(tmp_path: Path) -> None:
    source = tmp_path / "group.jpg"
    Image.new("RGB", (1600, 2000), (100, 120, 140)).save(source)
    other = [1050, 320, 1250, 570]
    record = CandidateRecord(
        source, "eligible", similarity=0.8, face_count=2, scale_bin="far", pose_bin="frontal",
        metadata={"target_bbox": [700, 300, 900, 550], "other_face_bboxes": [other]},
    )

    rows = prepare_dataset(tmp_path / "result", [record], FakeParser(), max_side=768, body_backend=FakeBody((1600, 2000)))

    assert len(rows) == 1
    crop_box = tuple(map(int, str(rows[0]["crop_box"]).split(",")))
    assert crop_box[2] < other[0]
    assert rows[0]["excluded_face_count"] == 1
    assert "other_faces_excluded" in str(rows[0]["crop_reasons"])
    assert (rows[0]["output_width"], rows[0]["output_height"]) in {
        (512, 512), (512, 768), (512, 1024), (768, 512), (768, 768),
        (768, 1024), (1024, 512), (1024, 768), (1024, 1024),
    }


def test_prepare_dataset_does_not_write_non_bucket_crop(tmp_path: Path) -> None:
    source = tmp_path / "small.jpg"
    Image.new("RGB", (400, 400), (100, 120, 140)).save(source)
    record = CandidateRecord(
        source, "eligible", similarity=0.8, scale_bin="far", pose_bin="frontal",
        metadata={"target_bbox": [150, 80, 250, 200]},
    )

    skips: list[CropSkip] = []
    rows = prepare_dataset(tmp_path / "result", [record], FakeParser(), body_backend=FakeBody((400, 400)), skips=skips)

    assert rows == []
    assert [(skip.rank, skip.record, skip.reason) for skip in skips] == [(1, record, "no_compatible_bucket")]
    assert list((tmp_path / "result" / "prepared").iterdir()) == []


def test_tighter_crop_skips_body_analysis_and_marks_identity_only(tmp_path: Path) -> None:
    source = tmp_path / "tight.jpg"
    Image.new("RGB", (1600, 2000), (100, 120, 140)).save(source)
    record = CandidateRecord(
        source, "eligible", similarity=0.8, scale_bin="far", pose_bin="frontal",
        metadata={"target_bbox": [700, 300, 900, 550]},
    )

    class UnexpectedBody:
        def analyze(self, *args):
            raise AssertionError("tight identity crop must not run body analysis")

    rows = prepare_dataset(
        tmp_path / "result",
        [record],
        FakeParser(),
        body_backend=UnexpectedBody(),
        crop_modes={source: "tight"},
    )

    assert len(rows) == 1
    assert rows[0]["strategy"].startswith("identity_tight")
    assert "identity_only" in str(rows[0]["crop_reasons"])
