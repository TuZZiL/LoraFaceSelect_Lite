from pathlib import Path

import numpy as np
import pytest

from lora_face_select_lite.backend import _pose_from_landmarks
from lora_face_select_lite.metrics import classify_pose, cosine_similarity, normalized_mean_embedding
from lora_face_select_lite.models import CandidateRecord
from lora_face_select_lite.selection import select_candidates


def test_embedding_normalization_and_cosine() -> None:
    value = normalized_mean_embedding([np.array([100.0, 0.0]), np.array([0.0, 1.0])])
    assert np.allclose(value, np.array([1.0, 1.0]) / np.sqrt(2))
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_embedding_normalization_rejects_zero_reference() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        normalized_mean_embedding([np.array([0.0, 0.0])])


def test_pose_categories_match_mvp_contract() -> None:
    assert classify_pose(0, -35) == "frontal"
    assert classify_pose(0, 35) == "frontal"
    assert classify_pose(-45, 0) == "left_profile"
    assert classify_pose(20, 0) == "right_three_quarter"


def test_five_landmark_pose_estimate_is_available() -> None:
    frontal = np.asarray([[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], dtype=np.float32)
    turned = np.asarray([[30, 30], [70, 30], [80, 50], [35, 70], [65, 70]], dtype=np.float32)

    yaw, pitch, roll = _pose_from_landmarks(frontal, (100, 100))
    turned_yaw, _, _ = _pose_from_landmarks(turned, (100, 100))

    assert yaw == pytest.approx(0)
    assert pitch == pytest.approx(0)
    assert roll == pytest.approx(0)
    assert turned_yaw is not None and turned_yaw >= 40


def test_selection_covers_scale_and_pose() -> None:
    records = [
        CandidateRecord(Path("a.jpg"), "eligible", similarity=.9, quality_score=.8, scale_bin="close", pose_bin="frontal", image_hash=0),
        CandidateRecord(Path("b.jpg"), "eligible", similarity=.8, quality_score=.8, scale_bin="far", pose_bin="left_profile", image_hash=(1 << 64) - 1),
        CandidateRecord(Path("c.jpg"), "eligible", similarity=.7, quality_score=.8, scale_bin="medium", pose_bin="right_profile", image_hash=0xAAAAAAAAAAAAAAAA),
    ]
    selected = select_candidates(records, 3, .35)
    assert {item.path for item in selected} == {Path("a.jpg"), Path("b.jpg"), Path("c.jpg")}


def test_selection_rejects_perceptual_duplicates() -> None:
    records = [
        CandidateRecord(Path("best.jpg"), "eligible", similarity=.9, quality_score=.8, image_hash=0),
        CandidateRecord(Path("duplicate.jpg"), "eligible", similarity=.8, quality_score=.8, image_hash=1),
    ]
    assert [item.path for item in select_candidates(records, 2, .35)] == [Path("best.jpg")]


def test_selection_accepts_eligible_group_photo() -> None:
    group = CandidateRecord(
        Path("group.jpg"), "eligible", similarity=.9, quality_score=.8,
        face_count=3, image_hash=0,
    )

    assert select_candidates([group], 1, .35) == [group]


def test_selection_reserves_thirty_percent_for_strong_body_shots() -> None:
    identity = [
        CandidateRecord(
            Path(f"identity_{index}.jpg"), "eligible", similarity=.9, quality_score=.9,
            face_count=1, scale_bin="medium", body_confidence=.9, image_hash=index,
        )
        for index in range(20)
    ]
    body = [
        CandidateRecord(
            Path(f"body_{index}.jpg"), "eligible", similarity=.6, quality_score=.8,
            face_count=1, scale_bin="far", body_confidence=.9, image_hash=100 + index,
        )
        for index in range(8)
    ]

    selected = select_candidates([*identity, *body], 20, .5, duplicate_threshold=-1)

    assert sum(record.body_focused for record in selected) >= 6


def test_group_photo_does_not_fill_body_quota() -> None:
    group = CandidateRecord(
        Path("group.jpg"), "eligible", similarity=.9, face_count=2,
        scale_bin="far", body_confidence=.9,
    )

    assert not group.body_focused


def test_selection_backfill_preserves_initial_and_excludes_failed_paths() -> None:
    initial = CandidateRecord(Path("initial.jpg"), "eligible", similarity=.9, image_hash=0)
    failed = CandidateRecord(Path("failed.jpg"), "eligible", similarity=.85, image_hash=1)
    replacement = CandidateRecord(Path("replacement.jpg"), "eligible", similarity=.8, image_hash=(1 << 64) - 1)

    selected = select_candidates(
        [initial, failed, replacement],
        2,
        .5,
        initial_selected=[initial],
        excluded_paths={str(failed.path)},
    )

    assert selected == [initial, replacement]


def test_selection_order_is_deterministic_for_ties() -> None:
    records = [
        CandidateRecord(Path("b.jpg"), "eligible", similarity=.8, quality_score=.8, image_hash=0),
        CandidateRecord(Path("a.jpg"), "eligible", similarity=.8, quality_score=.8, image_hash=(1 << 64) - 1),
    ]
    first = [item.path for item in select_candidates(records, 2, .35)]
    second = [item.path for item in select_candidates(list(reversed(records)), 2, .35)]
    assert first == second


def test_selection_prefers_reference_consistency_over_age_diversity() -> None:
    reference_period = CandidateRecord(
        Path("reference_period.jpg"), "eligible", similarity=.65, quality_score=.85,
        face_width_ratio=.11, scale_bin="medium", pose_bin="frontal", yaw=0,
        image_hash=0,
    )
    younger_clear_face = CandidateRecord(
        Path("younger.jpg"), "eligible", similarity=.46, quality_score=.88,
        face_width_ratio=.21, scale_bin="medium", pose_bin="frontal", yaw=10,
        image_hash=(1 << 64) - 1,
    )

    selected = select_candidates([reference_period, younger_clear_face], 1, .35)

    assert selected == [reference_period]


def test_appearance_mismatch_beats_unique_scale_bonus() -> None:
    hashes = (0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA)
    matching = [
        CandidateRecord(
            Path(f"matching_{index}.jpg"), "eligible", similarity=.60,
            appearance_similarity=.84, quality_score=.8, face_width_ratio=.12,
            scale_bin="medium", pose_bin="frontal", yaw=0, image_hash=hashes[index],
        )
        for index in range(3)
    ]
    mismatched_close = CandidateRecord(
        Path("mismatched_close.jpg"), "eligible", similarity=.62,
        appearance_similarity=.67, quality_score=.85, face_width_ratio=.25,
        scale_bin="close", pose_bin="frontal", yaw=0, image_hash=(1 << 64) - 1,
    )
    selected = select_candidates([*matching, mismatched_close], 3, .50)
    assert mismatched_close not in selected
