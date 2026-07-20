from pathlib import Path

import cv2
import numpy as np
import pytest

from lora_face_select_lite.body_attributes import estimate_body_attributes
from lora_face_select_lite.nudenet import LABELS, NudeNetBackend


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.inputs: list[np.ndarray] = []

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert output_names == ["output0"]
        assert list(input_feed) == ["images"]
        self.inputs.append(input_feed["images"])
        return [self.output]


def _backend(output: np.ndarray) -> NudeNetBackend:
    backend = NudeNetBackend.__new__(NudeNetBackend)
    backend.score_threshold = 0.25
    backend.nms_threshold = 0.45
    backend.input_size = 320
    backend.model_path = Path("fake.onnx")
    backend.session = FakeSession(output)
    backend.input_name = "images"
    backend.output_name = "output0"
    backend.provider = "test"
    return backend


def test_yolov8_output_is_decoded_and_mapped_back_from_target_roi() -> None:
    output = np.zeros((1, 4 + len(LABELS), 2), dtype=np.float32)
    # ROI (40, 20, 160, 180) is expanded and becomes 148x176 at (26, 12).
    # A centered 64x64 model-space box must stay inside that target-person ROI.
    output[0, :4, 0] = (160, 160, 64, 64)
    output[0, 4 + 3, 0] = 0.90
    backend = _backend(output)

    result = backend.detect(np.zeros((200, 200, 3), dtype=np.uint8), roi=(40, 20, 160, 180))

    assert result.labels == ["female-breast-exposed"]
    assert result.max_detection_score == pytest.approx(0.90)
    detection = result.detections[0]
    assert detection.box[0] >= 26
    assert detection.box[1] >= 12
    assert detection.box[0] + detection.box[2] <= 174
    assert detection.box[1] + detection.box[3] <= 188
    assert backend.session.inputs[0].shape == (1, 3, 320, 320)


def test_unexpected_model_contract_fails_loudly() -> None:
    backend = _backend(np.zeros((1, 20, 100), dtype=np.float32))
    with pytest.raises(RuntimeError, match="Unexpected NudeNet output shape"):
        backend.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_empty_or_invalid_roi_returns_no_detections() -> None:
    backend = _backend(np.zeros((1, 4 + len(LABELS), 1), dtype=np.float32))
    result = backend.detect(np.zeros((100, 100, 3), dtype=np.uint8), roi=(200, 200, 250, 250))
    assert result.detections == []
    assert backend.session.inputs == []


class MisleadingNudeNetResult:
    def max_score(self, label: str) -> float:
        return 1.0


def test_nudenet_confidence_does_not_change_anatomical_estimates() -> None:
    landmarks = np.zeros((33, 5), dtype=np.float32)
    for index, point in {11: (30, 30), 12: (70, 30), 23: (35, 80), 24: (65, 80)}.items():
        landmarks[index] = (*point, 0, 1, 1)
    without_nudenet = estimate_body_attributes(landmarks, 0.9, face_bbox=(40, 5, 60, 25))
    with_nudenet = estimate_body_attributes(
        landmarks,
        0.9,
        face_bbox=(40, 5, 60, 25),
        nudenet_detections=MisleadingNudeNetResult(),
    )
    assert with_nudenet == without_nudenet


def test_real_model_initializes_and_matches_expected_contract() -> None:
    model = Path("models/stable/nudenet_320n.onnx")
    if not model.is_file():
        pytest.skip("NudeNet model is optional")
    backend = NudeNetBackend(model)
    output = backend.session.run(
        [backend.output_name],
        {backend.input_name: np.zeros((1, 3, 320, 320), dtype=np.float32)},
    )[0]
    assert output.shape[1:] == (4 + len(LABELS), 2100)
