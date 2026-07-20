from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .body_attributes import estimate_body_attributes
from .io_utils import image_dimensions, read_image
from .metrics import classify_lighting, classify_pose, classify_scale, cosine_similarity, image_hash, image_quality, normalized_mean_embedding
from .models import CandidateRecord


def reference_embedding(reference_paths: list[Path], backend: Any, max_side: int = 960) -> Any:
    embeddings = []
    for path in reference_paths:
        image = read_image(path, max_side=max_side)
        if image is None:
            raise ValueError(f"Cannot read reference image: {path}")
        faces = backend.analyze(image)
        if len(faces) != 1:
            raise ValueError(f"Reference must contain exactly one face: {path} (found {len(faces)})")
        embeddings.append(faces[0].embedding)
    return normalized_mean_embedding(embeddings)


def reference_appearance(reference_paths: list[Path], face_backend: Any, appearance_backend: Any, max_side: int = 960) -> tuple[Any, Any]:
    face_embeddings, head_embeddings = [], []
    for path in reference_paths:
        image = read_image(path, max_side=max_side)
        if image is None:
            raise ValueError(f"Cannot read reference image: {path}")
        faces = face_backend.analyze(image)
        if len(faces) != 1:
            raise ValueError(f"Reference must contain exactly one face: {path} (found {len(faces)})")
        face_embedding, head_embedding = appearance_backend.embed(image, faces[0].bbox)
        face_embeddings.append(face_embedding)
        head_embeddings.append(head_embedding)
    return normalized_mean_embedding(face_embeddings), normalized_mean_embedding(head_embeddings)


ProgressCallback = Callable[[int, int, Path], None]


def _expanded_bbox(bbox: list[float], padding_ratio: float) -> tuple[float, float, float, float]:
    padding = max(4.0, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * padding_ratio)
    return bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding


def _boxes_intersect(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def analyze_dataset(
    paths: list[Path],
    identity: Any,
    backend: Any,
    min_similarity: float,
    min_face_width: int = 48,
    min_quality: float = 0.15,
    max_abs_yaw: float | None = 25.0,
    max_side: int = 960,
    progress: ProgressCallback | None = None,
    appearance_identity: tuple[Any, Any] | None = None,
    appearance_backend: Any | None = None,
    body_backend: Any | None = None,
    nudenet_backend: Any | None = None,
    min_body_pose_confidence: float = 0.50,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    readable_images = backend_errors = 0
    for current, path in enumerate(paths, 1):
        if progress is not None:
            progress(current, len(paths), path)
        image = read_image(path, max_side=max_side)
        if image is None:
            records.append(CandidateRecord(path=path, status="rejected", reason="unreadable"))
            continue
        readable_images += 1
        height, width = image.shape[:2]
        try:
            faces = backend.analyze(image)
        except Exception as exc:
            backend_errors += 1
            records.append(CandidateRecord(path=path, status="rejected", reason=f"backend_error: {exc}"))
            continue
        if not faces:
            records.append(CandidateRecord(path=path, status="rejected", reason="no_face"))
            continue
        similarity, target_index, target = max(
            ((cosine_similarity(face.embedding, identity), index, face) for index, face in enumerate(faces)),
            key=lambda item: item[0],
        )
        appearance_face = appearance_head = appearance = None
        x1, y1, x2, y2 = target.bbox
        face_width = max(0.0, min(float(width), x2) - max(0.0, x1))
        ratio = face_width / max(1, width)
        original_size = image_dimensions(path) or (width, height)
        resolution_scale = original_size[0] / max(1, width)
        raw = image_quality(image, target.bbox, resolution_scale=resolution_scale)
        blur_score = max(0.0, min(1.0, raw["blur_raw"] / 8.0))
        exposure_score = max(0.0, min(1.0, raw["exposure_raw"]))
        resolution_score = max(0.0, min(1.0, raw["resolution_raw"]))
        margin_score = max(0.0, min(1.0, raw["margin_raw"]))
        quality_score = 0.40 * blur_score + 0.25 * exposure_score + 0.20 * resolution_score + 0.15 * margin_score
        similarity_ok = similarity >= min_similarity
        size_ok = face_width >= min_face_width and ratio >= 0.04
        pose_ok = max_abs_yaw is None or target.yaw is None or abs(target.yaw) <= max_abs_yaw
        # A weighted average alone can hide a catastrophically blurred or
        # clipped face behind good resolution/margins, so keep modest floors
        # for the two critical visual signals as well.
        quality_ok = (
            quality_score >= min_quality
            and blur_score >= min_quality * 0.25
            and exposure_score >= min_quality * 0.50
        )
        candidate_analysis_ok = similarity >= min_similarity - 0.15 and size_ok and quality_ok and pose_ok
        if candidate_analysis_ok and appearance_identity is not None and appearance_backend is not None:
            try:
                face_embedding, head_embedding = appearance_backend.embed(image, target.bbox)
                appearance_face = cosine_similarity(face_embedding, appearance_identity[0])
                appearance_head = cosine_similarity(head_embedding, appearance_identity[1])
                appearance = 0.65 * appearance_face + 0.35 * appearance_head
            except Exception:
                # Identity/quality analysis remains useful if the optional
                # appearance model cannot process one unusual crop.
                pass
        body_attrs = None
        nudenet_result = None
        nudenet_error = None
        if candidate_analysis_ok and body_backend is not None:
            body = body_backend.analyze(image, face_bbox=(x1, y1, x2, y2))
            if body is not None and body.confidence >= min_body_pose_confidence:
                if nudenet_backend is not None:
                    try:
                        nudenet_result = nudenet_backend.detect(image, roi=body.bbox)
                    except Exception as exc:
                        nudenet_error = f"{type(exc).__name__}: {exc}"
                body_attrs = estimate_body_attributes(body.landmarks, body.confidence, face_bbox=(x1, y1, x2, y2), nudenet_detections=nudenet_result)
        status = "eligible" if similarity_ok and size_ok and quality_ok and pose_ok else "rejected"
        if not similarity_ok:
            reason = "below_similarity"
        elif not size_ok:
            reason = "face_too_small"
        elif blur_score < min_quality * 0.25:
            reason = "low_quality_blur"
        elif exposure_score < min_quality * 0.50:
            reason = "low_quality_exposure"
        elif not quality_ok:
            reason = "low_quality"
        elif not pose_ok:
            reason = "pose_too_extreme"
        else:
            reason = ""
        scale_x, scale_y = original_size[0] / width, original_size[1] / height
        target_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        other_face_bboxes = [
            [face.bbox[0] * scale_x, face.bbox[1] * scale_y, face.bbox[2] * scale_x, face.bbox[3] * scale_y]
            for index, face in enumerate(faces)
            if index != target_index
        ]
        record = CandidateRecord(
            path=path,
            status=status,
            reason=reason,
            similarity=similarity,
            appearance_similarity=appearance,
            appearance_face_similarity=appearance_face,
            appearance_head_similarity=appearance_head,
            face_count=len(faces),
            face_width_ratio=ratio,
            scale_bin=classify_scale(ratio),
            pose_bin=classify_pose(target.yaw, target.pitch),
            lighting_bin=classify_lighting(raw["mean_luma"]),
            yaw=target.yaw,
            pitch=target.pitch,
            roll=target.roll,
            quality_score=quality_score,
            blur_score=blur_score,
            exposure_score=exposure_score,
            resolution_score=resolution_score,
            margin_score=margin_score,
            image_hash=image_hash(image),
            detection_score=target.detection_score,
            body_shape=None if body_attrs is None else body_attrs.body_shape,
            chest_estimate=None if body_attrs is None else body_attrs.chest_estimate,
            build=None if body_attrs is None else body_attrs.build,
            shoulder_hip_ratio=None if body_attrs is None else body_attrs.shoulder_hip_ratio,
            torso_proportion=None if body_attrs is None else body_attrs.torso_proportion,
            body_confidence=None if body_attrs is None else body_attrs.confidence,
            nudenet_labels=None if nudenet_result is None else ";".join(nudenet_result.labels),
            nudenet_max_score=None if nudenet_result is None else nudenet_result.max_detection_score,
            nudenet_error=nudenet_error,
            metadata={
                "width": original_size[0],
                "height": original_size[1],
                "analysis_width": width,
                "analysis_height": height,
                "target_bbox": target_bbox,
                "other_face_bboxes": other_face_bboxes,
            },
        )
        record.fallback_eligible = quality_ok and size_ok and pose_ok and len(faces) == 1 and similarity >= min_similarity - 0.15
        target_guard = _expanded_bbox(target_bbox, 0.08)
        if record.eligible and any(_boxes_intersect(_expanded_bbox(other, 0.40), target_guard) for other in other_face_bboxes):
            record.status = "multiple_faces_review"
            record.reason = "faces_overlap"
        records.append(record)
    if readable_images and backend_errors == readable_images:
        raise RuntimeError("Face backend failed for every readable image; check the model files and OpenCV.")
    return records
