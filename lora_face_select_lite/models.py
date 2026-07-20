from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FaceObservation:
    embedding: Any
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    detection_score: float = 0.0
    landmarks: Any = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


@dataclass
class CandidateRecord:
    path: Path
    status: str
    reason: str = ""
    similarity: float | None = None
    appearance_similarity: float | None = None
    appearance_face_similarity: float | None = None
    appearance_head_similarity: float | None = None
    face_count: int = 0
    face_width_ratio: float | None = None
    scale_bin: str = "unknown"
    pose_bin: str = "unknown"
    lighting_bin: str = "unknown"
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    quality_score: float = 0.0
    blur_score: float = 0.0
    exposure_score: float = 0.0
    resolution_score: float = 0.0
    margin_score: float = 0.0
    image_hash: int | None = None
    fallback_eligible: bool = False
    detection_score: float | None = None
    body_shape: str | None = None
    chest_estimate: str | None = None
    build: str | None = None
    shoulder_hip_ratio: float | None = None
    torso_proportion: float | None = None
    body_confidence: float | None = None
    nudenet_labels: str | None = None
    nudenet_max_score: float | None = None
    nudenet_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"

    @property
    def body_focused(self) -> bool:
        return self.face_count == 1 and self.scale_bin == "far" and self.body_confidence is not None
