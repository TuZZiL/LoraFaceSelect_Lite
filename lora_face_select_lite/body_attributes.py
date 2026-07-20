"""Body attribute estimation from MediaPipe pose landmarks.

Computes body shape, chest estimate, and build classification from
33-point pose landmarks. Results are proportional (scale-invariant)
and require reliable shoulder and hip keypoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BodyAttributes:
    body_shape: str
    chest_estimate: str
    build: str
    shoulder_hip_ratio: float | None
    torso_proportion: float | None
    confidence: float


def _keypoint_visible(landmarks: Any, index: int) -> bool:
    return bool(landmarks[index, 3] >= 0.35 and landmarks[index, 4] >= 0.60)


def estimate_body_attributes(
    landmarks: Any,
    pose_confidence: float,
    face_bbox: tuple[float, float, float, float] | None = None,
    nudenet_detections: Any | None = None,
) -> BodyAttributes:
    import numpy as np

    shoulder_indices = (11, 12)
    hip_indices = (23, 24)

    shoulders_ok = all(_keypoint_visible(landmarks, i) for i in shoulder_indices)
    hips_ok = all(_keypoint_visible(landmarks, i) for i in hip_indices)

    if not shoulders_ok and not hips_ok:
        return BodyAttributes("unknown", "unknown", "unknown", None, None, pose_confidence)

    def _dist(a: int, b: int) -> float:
        return float(np.linalg.norm(landmarks[a, :2] - landmarks[b, :2]))

    shoulder_width: float | None = _dist(11, 12) if shoulders_ok else None
    hip_width: float | None = _dist(23, 24) if hips_ok else None

    shoulder_hip_ratio = None
    if shoulder_width is not None and hip_width is not None and hip_width > 1.0:
        shoulder_hip_ratio = shoulder_width / hip_width

    shoulder_mid = (landmarks[11, :2] + landmarks[12, :2]) / 2.0 if shoulders_ok else None
    hip_mid = (landmarks[23, :2] + landmarks[24, :2]) / 2.0 if hips_ok else None

    torso_proportion = None
    if shoulder_mid is not None and hip_mid is not None and shoulder_width is not None and shoulder_width > 1.0:
        torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))
        torso_proportion = torso_height / shoulder_width

    face_width: float | None = None
    if face_bbox is not None:
        face_width = max(1.0, face_bbox[2] - face_bbox[0])

    body_shape = _classify_body_shape(shoulder_hip_ratio)
    # NudeNet confidence measures detection certainty, not anatomical size.
    # Keep body/chest estimates based only on pose geometry.
    chest_estimate = _estimate_chest(shoulder_width, face_width, torso_proportion, shoulder_hip_ratio, pose_confidence)
    build = _classify_build(shoulder_hip_ratio, torso_proportion, pose_confidence)

    return BodyAttributes(body_shape, chest_estimate, build, shoulder_hip_ratio, torso_proportion, pose_confidence)


def _classify_body_shape(shoulder_hip_ratio: float | None) -> str:
    if shoulder_hip_ratio is None:
        return "unknown"
    if 0.93 <= shoulder_hip_ratio <= 1.07:
        return "hourglass"
    if shoulder_hip_ratio < 0.90:
        return "pear"
    if shoulder_hip_ratio > 1.10:
        return "inverted_triangle"
    return "rectangle"


def _estimate_chest(
    shoulder_width: float | None,
    face_width: float | None,
    torso_proportion: float | None,
    shoulder_hip_ratio: float | None,
    pose_confidence: float,
    nudenet_detections: Any | None = None,
) -> str:
    if shoulder_width is None or pose_confidence < 0.40:
        return "unknown"

    score = 0.0
    signals = 0

    if face_width is not None:
        sw_to_face = shoulder_width / face_width
        signals += 1
        if sw_to_face > 3.2:
            score += 1.0
        elif sw_to_face > 2.6:
            score += 0.5

    if torso_proportion is not None:
        signals += 1
        if torso_proportion < 0.55:
            score += 0.7
        elif torso_proportion < 0.62:
            score += 0.4
        else:
            score += 0.1

    if shoulder_hip_ratio is not None:
        signals += 1
        if shoulder_hip_ratio > 1.02:
            score += 0.8
        elif shoulder_hip_ratio > 0.95:
            score += 0.4
        else:
            score += 0.1

    if signals == 0:
        return "unknown"

    avg = score / signals
    if avg > 0.55:
        return "large"
    if avg > 0.28:
        return "medium"
    return "small"


def _classify_build(
    shoulder_hip_ratio: float | None,
    torso_proportion: float | None,
    pose_confidence: float,
    nudenet_detections: Any | None = None,
) -> str:
    if shoulder_hip_ratio is None and torso_proportion is None:
        return "unknown"
    if pose_confidence < 0.40:
        return "unknown"

    curvy_score = 0
    athletic_score = 0
    slim_score = 0
    signals = 0

    if shoulder_hip_ratio is not None:
        signals += 1
        if shoulder_hip_ratio < 0.92:
            curvy_score += 1
        elif shoulder_hip_ratio > 1.08:
            athletic_score += 1
        elif shoulder_hip_ratio < 0.98:
            curvy_score += 0.3

    if torso_proportion is not None:
        signals += 1
        if torso_proportion > 0.65:
            slim_score += 1
        elif torso_proportion > 0.58:
            slim_score += 0.5
        if torso_proportion < 0.53:
            athletic_score += 0.7

    if signals == 0:
        return "unknown"

    scores = {"curvy": curvy_score / signals, "athletic": athletic_score / signals, "slim": slim_score / signals}
    best = max(scores, key=scores.get)
    if scores[best] > 0.4:
        return best
    return "average"
