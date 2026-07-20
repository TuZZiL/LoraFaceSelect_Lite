import numpy as np

from lora_face_select_lite.body import MediaPipeBodyBackend, _anchors


def test_mediapipe_anchor_generation_matches_model_contract() -> None:
    anchors = _anchors()
    assert anchors.shape == (2254, 2)
    assert np.allclose(anchors[0], (0.5 / 28, 0.5 / 28))
    assert np.allclose(anchors[-1], (6.5 / 7, 6.5 / 7))


def _person(face_box, score=0.9):
    return np.asarray([*face_box, *([0.0] * 8), score], dtype=np.float32)


def test_person_matching_uses_target_face_not_highest_global_score() -> None:
    target = _person((100, 100, 200, 220), 0.8)
    stranger = _person((600, 100, 700, 220), 0.99)
    selected = MediaPipeBodyBackend._match_person([stranger, target], (110, 110, 195, 215))
    assert selected is target


def test_person_matching_rejects_distant_detection() -> None:
    stranger = _person((600, 100, 700, 220))
    assert MediaPipeBodyBackend._match_person([stranger], (100, 100, 200, 220)) is None
