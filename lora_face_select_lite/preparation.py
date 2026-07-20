from __future__ import annotations

import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_image
from .parsing import label_ratios


HEAD_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18}
PALETTE = (
    (0, 0, 0), (255, 204, 180), (128, 64, 128), (128, 64, 128),
    (80, 160, 255), (80, 160, 255), (40, 40, 40), (255, 180, 140),
    (255, 180, 140), (255, 220, 60), (255, 160, 120), (220, 80, 100),
    (255, 80, 120), (180, 40, 80), (180, 140, 100), (255, 215, 0),
    (80, 180, 100), (100, 60, 30), (90, 100, 200),
)

SKELETON = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29),
    (27, 31), (24, 26), (26, 28), (28, 30), (28, 32),
)
HAND_EDGES = {(15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)}


@dataclass(frozen=True)
class CropDecision:
    box: tuple[int, int, int, int]
    strategy: str
    safety: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CropSkip:
    rank: int
    record: Any
    reason: str


def _clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    return left, top, right, bottom


def _centered_box(width: int, height: int, cx: float, cy: float, wanted_w: float, wanted_h: float) -> tuple[int, int, int, int]:
    wanted_w, wanted_h = min(float(width), wanted_w), min(float(height), wanted_h)
    left = min(max(0.0, cx - wanted_w / 2), width - wanted_w)
    top = min(max(0.0, cy - wanted_h / 2), height - wanted_h)
    return _clamp_box((left, top, left + wanted_w, top + wanted_h), width, height)


def _boxes_intersect(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def _intersection(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return max(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), min(first[3], second[3])


def face_safe_bounds(
    image_size: tuple[int, int],
    target_bbox: tuple[float, float, float, float],
    other_face_bboxes: list[tuple[float, float, float, float]],
    preferred_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Return largest useful rectangle containing target but no other detected face."""
    width, height = image_size
    face_size = max(target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1])
    target_padding = max(4.0, face_size * 0.08)
    guard = (
        max(0.0, target_bbox[0] - target_padding),
        max(0.0, target_bbox[1] - target_padding),
        min(float(width), target_bbox[2] + target_padding),
        min(float(height), target_bbox[3] + target_padding),
    )
    candidates = [(0, 0, width, height)]
    for other in other_face_bboxes:
        other_size = max(other[2] - other[0], other[3] - other[1])
        head_padding = max(4.0, other_size * 0.40)
        obstacle = (
            other[0] - head_padding,
            other[1] - head_padding,
            other[2] + head_padding,
            other[3] + head_padding,
        )
        if _boxes_intersect(guard, obstacle):
            return None
        next_candidates: list[tuple[int, int, int, int]] = []
        for bounds in candidates:
            if not _boxes_intersect(bounds, obstacle):
                next_candidates.append(bounds)
                continue
            left, top, right, bottom = bounds
            if obstacle[2] <= guard[0]:
                next_candidates.append((max(left, math.ceil(obstacle[2])), top, right, bottom))
            if obstacle[0] >= guard[2]:
                next_candidates.append((left, top, min(right, math.floor(obstacle[0])), bottom))
            if obstacle[3] <= guard[1]:
                next_candidates.append((left, max(top, math.ceil(obstacle[3])), right, bottom))
            if obstacle[1] >= guard[3]:
                next_candidates.append((left, top, right, min(bottom, math.floor(obstacle[1]))))
        valid_candidates = [
            box for box in next_candidates
            if box[0] <= guard[0] and box[1] <= guard[1] and box[2] >= guard[2] and box[3] >= guard[3]
        ]
        candidates = [
            box for box in dict.fromkeys(valid_candidates)
            if not any(
                other != box
                and other[0] <= box[0] and other[1] <= box[1]
                and other[2] >= box[2] and other[3] >= box[3]
                for other in valid_candidates
            )
        ]
        if not candidates:
            return None

    def score(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
        overlap = _intersection(bounds, preferred_box)
        overlap_area = max(0, overlap[2] - overlap[0]) * max(0, overlap[3] - overlap[1])
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        return overlap_area, area

    return max(candidates, key=score)


def _centered_box_in_bounds(
    bounds: tuple[int, int, int, int], cx: float, cy: float, wanted_w: int, wanted_h: int,
) -> tuple[int, int, int, int]:
    left = min(max(float(bounds[0]), cx - wanted_w / 2), bounds[2] - wanted_w)
    top = min(max(float(bounds[1]), cy - wanted_h / 2), bounds[3] - wanted_h)
    return round(left), round(top), round(left + wanted_w), round(top + wanted_h)


def _box_in_bounds_containing(
    bounds: tuple[int, int, int, int],
    required: tuple[float, float, float, float],
    cx: float,
    cy: float,
    wanted_w: int,
    wanted_h: int,
) -> tuple[int, int, int, int] | None:
    min_left = max(bounds[0], math.ceil(required[2] - wanted_w))
    max_left = min(bounds[2] - wanted_w, math.floor(required[0]))
    min_top = max(bounds[1], math.ceil(required[3] - wanted_h))
    max_top = min(bounds[3] - wanted_h, math.floor(required[1]))
    if min_left > max_left or min_top > max_top:
        return None
    left = min(max(round(cx - wanted_w / 2), min_left), max_left)
    top = min(max(round(cy - wanted_h / 2), min_top), max_top)
    return left, top, left + wanted_w, top + wanted_h


def fit_crop_to_training_bucket(
    box: tuple[int, int, int, int], image_size: tuple[int, int], min_side: int, max_side: int,
    bounds: tuple[int, int, int, int] | None = None,
    required_box: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[int, int, int, int], tuple[int, int] | None]:
    """Expand a safe crop to the closest exact Krea 2 training bucket."""
    width, height = image_size
    bounds = bounds or (0, 0, width, height)
    bounds_w, bounds_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x1, y1, x2, y2 = box
    crop_w, crop_h = x2 - x1, y2 - y1
    sides = tuple(side for side in (512, 768, 1024) if min_side <= side <= max_side)
    candidates: list[tuple[tuple[float, float, int], tuple[int, int, int, int], tuple[int, int]]] = []
    for output_w in sides:
        for output_h in sides:
            divisor = math.gcd(output_w, output_h)
            unit_w, unit_h = output_w // divisor, output_h // divisor
            # Integer multiples preserve the bucket aspect ratio exactly. The
            # divisor lower bound guarantees that resizing never upscales.
            multiplier = max(divisor, math.ceil(crop_w / unit_w), math.ceil(crop_h / unit_h))
            wanted_w, wanted_h = unit_w * multiplier, unit_h * multiplier
            if wanted_w > bounds_w or wanted_h > bounds_h:
                continue
            candidate = _centered_box_in_bounds(bounds, (x1 + x2) / 2, (y1 + y2) / 2, wanted_w, wanted_h)
            if not (candidate[0] <= x1 and candidate[1] <= y1 and candidate[2] >= x2 and candidate[3] >= y2):
                continue
            aspect_error = abs(math.log((output_w / output_h) / (crop_w / crop_h)))
            expansion = (wanted_w * wanted_h) / max(1, crop_w * crop_h)
            candidates.append(((aspect_error, expansion, -(output_w * output_h)), candidate, (output_w, output_h)))
    if not candidates:
        if required_box is None:
            return box, None
        # The preferred crop can be the whole source or a face-safe zone and
        # therefore impossible to expand. Trim context to the largest exact
        # bucket ratio while keeping the target face intact and never upscale.
        fallback_candidates: list[
            tuple[tuple[float, float, int, int], tuple[int, int, int, int], tuple[int, int]]
        ] = []
        preferred_area = max(1, crop_w * crop_h)
        for output_w in sides:
            for output_h in sides:
                divisor = math.gcd(output_w, output_h)
                unit_w, unit_h = output_w // divisor, output_h // divisor
                multiplier = min(bounds_w // unit_w, bounds_h // unit_h)
                if multiplier < divisor:
                    continue
                wanted_w, wanted_h = unit_w * multiplier, unit_h * multiplier
                candidate = _box_in_bounds_containing(
                    bounds, required_box, (x1 + x2) / 2, (y1 + y2) / 2, wanted_w, wanted_h,
                )
                if candidate is None:
                    continue
                overlap = _intersection(candidate, box)
                retained = max(0, overlap[2] - overlap[0]) * max(0, overlap[3] - overlap[1]) / preferred_area
                aspect_error = abs(math.log((output_w / output_h) / (crop_w / crop_h)))
                score = -retained, aspect_error, -(wanted_w * wanted_h), -(output_w * output_h)
                fallback_candidates.append((score, candidate, (output_w, output_h)))
        if not fallback_candidates:
            return box, None
        _, candidate, bucket = min(fallback_candidates, key=lambda item: item[0])
        return candidate, bucket
    _, candidate, bucket = min(candidates, key=lambda item: item[0])
    return candidate, bucket


def parsing_region(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    face = max(x2 - x1, y2 - y1)
    return _centered_box(width, height, (x1 + x2) / 2, (y1 + y2) / 2 - 0.12 * face, 2.8 * face, 2.8 * face)


def recommended_crop(
    image_size: tuple[int, int],
    bbox: tuple[float, float, float, float],
    scale_bin: str,
    semantic_box: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[int, int, int, int], str]:
    width, height = image_size
    if scale_bin == "far":
        return (0, 0, width, height), "context_original"
    x1, y1, x2, y2 = bbox
    face_h = max(1.0, y2 - y1)
    if semantic_box:
        sx1, sy1, sx2, sy2 = semantic_box
        cx = (sx1 + sx2) / 2
        top = min(sy1, y1 - 0.20 * face_h)
        bottom = max(sy2, y2 + (0.75 if scale_bin == "close" else 1.65) * face_h)
    else:
        cx, top = (x1 + x2) / 2, y1 - 0.35 * face_h
        bottom = y2 + (0.85 if scale_bin == "close" else 1.75) * face_h
    desired_h = max(bottom - top, face_h * (2.2 if scale_bin == "close" else 3.4))
    aspect = 4 / 5 if scale_bin == "close" else 3 / 4
    desired_w = desired_h * aspect
    cy = (top + bottom) / 2
    return _centered_box(width, height, cx, cy, desired_w, desired_h), f"smart_{scale_bin}"


def _inside(point: Any, box: tuple[int, int, int, int]) -> bool:
    return box[0] <= float(point[0]) <= box[2] and box[1] <= float(point[1]) <= box[3]


def _crop_risks(box: tuple[int, int, int, int], image_size: tuple[int, int], semantic_box: tuple[float, float, float, float] | None, body: Any) -> tuple[str, ...]:
    import numpy as np

    width, height = image_size
    x1, y1, x2, y2 = box
    crop_w, crop_h = x2 - x1, y2 - y1
    margin = max(8.0, min(crop_w, crop_h) * 0.045)
    risks: set[str] = set()
    if semantic_box:
        sx1, sy1, sx2, sy2 = semantic_box
        head_risk = (
            (x1 > 0 and sx1 < x1 + margin)
            or (y1 > 0 and sy1 < y1 + margin)
            or (x2 < width and sx2 > x2 - margin)
            or (y2 < height and sy2 > y2 - margin)
        )
        if head_risk:
            risks.add("head_near_edge")
    landmarks = body.landmarks
    reliable = (landmarks[:, 3] >= 0.35) & (landmarks[:, 4] >= 0.60)
    # MediaPipe exposes wrist + thumb/index/pinky points. A hand must be
    # wholly inside with margin or wholly outside; other limbs may be cropped.
    important = set(range(15, 23))
    for index in np.flatnonzero(reliable):
        point = landmarks[index, :2]
        if index in important and _inside(point, box):
            distances = (point[0] - x1, x2 - point[0], point[1] - y1, y2 - point[1])
            for edge, distance in enumerate(distances):
                at_source_edge = (edge == 0 and x1 == 0) or (edge == 1 and x2 == width) or (edge == 2 and y1 == 0) or (edge == 3 and y2 == height)
                if distance < margin and not at_source_edge:
                    risks.add("hand_near_edge")
    for elbow, wrist in ((13, 15), (14, 16)):
        if not reliable[wrist] or not _inside(landmarks[wrist], box):
            continue
        hand_margin = margin
        if reliable[elbow]:
            hand_margin = max(hand_margin, float(np.linalg.norm(landmarks[wrist, :2] - landmarks[elbow, :2])) * 0.35)
        point = landmarks[wrist, :2]
        distances = (point[0] - x1, x2 - point[0], point[1] - y1, y2 - point[1])
        for edge, distance in enumerate(distances):
            at_source_edge = (edge == 0 and x1 == 0) or (edge == 1 and x2 == width) or (edge == 2 and y1 == 0) or (edge == 3 and y2 == height)
            if distance < hand_margin and not at_source_edge:
                risks.add("hand_near_edge")
    for first, second in SKELETON:
        if not (reliable[first] and reliable[second]):
            continue
        first_inside, second_inside = _inside(landmarks[first], box), _inside(landmarks[second], box)
        if first_inside == second_inside:
            continue
        outside = landmarks[second if first_inside else first, :2]
        if (first, second) in HAND_EDGES:
            risks.add("fingers_cut")
    return tuple(sorted(risks))


def _soft_joint_risk(box: tuple[int, int, int, int], image_size: tuple[int, int], body: Any) -> bool:
    """Prefer not to cut at elbows/knees, but never reject a safe hand crop."""
    import numpy as np

    width, height = image_size
    x1, y1, x2, y2 = box
    margin = max(8.0, min(x2 - x1, y2 - y1) * 0.035)
    landmarks = body.landmarks
    reliable = (landmarks[:, 3] >= 0.35) & (landmarks[:, 4] >= 0.60)
    for index in (13, 14, 23, 24, 25, 26, 27, 28):
        if not reliable[index] or not _inside(landmarks[index], box):
            continue
        point = landmarks[index, :2]
        distances = (point[0] - x1, x2 - point[0], point[1] - y1, y2 - point[1])
        for edge, distance in enumerate(distances):
            at_source_edge = (edge == 0 and x1 == 0) or (edge == 1 and x2 == width) or (edge == 2 and y1 == 0) or (edge == 3 and y2 == height)
            if distance < margin and not at_source_edge:
                return True
    return False


def body_aware_crop(
    image_size: tuple[int, int],
    bbox: tuple[float, float, float, float],
    scale_bin: str,
    semantic_box: tuple[float, float, float, float] | None,
    body: Any | None,
    min_pose_confidence: float = 0.50,
) -> CropDecision:
    width, height = image_size
    original = (0, 0, width, height)
    if scale_bin == "far":
        if body is not None and hasattr(body, "bbox") and body.confidence >= min_pose_confidence:
            bx1, by1, bx2, by2 = body.bbox
            bw = bx2 - bx1
            bh = by2 - by1
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            padding_x = max(16.0, bw * 0.15)
            padding_y = max(16.0, bh * 0.10)
            desired_h = bh + 2.0 * padding_y
            desired_w = desired_h * 0.75
            if desired_w < (bw + 2.0 * padding_x):
                desired_w = bw + 2.0 * padding_x
                desired_h = desired_w / 0.75
            candidate = _centered_box(width, height, cx, cy, desired_w, desired_h)
            risks = _crop_risks(candidate, image_size, semantic_box, body)
            if not risks:
                saved_area = 1.0 - ((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])) / max(1, width * height)
                if saved_area >= 0.08:
                    return CropDecision(candidate, "body_crop_far", "safe", ())
        return CropDecision(original, "context_original", "safe_original", ("far_view_preserved",))
    if body is None:
        return CropDecision(original, "original_fallback", "warning", ("body_not_found",))
    if body.confidence < min_pose_confidence:
        return CropDecision(original, "original_fallback", "warning", ("pose_uncertain",))
    base, base_strategy = recommended_crop(image_size, bbox, scale_bin, semantic_box)
    cx, cy = (base[0] + base[2]) / 2, (base[1] + base[3]) / 2
    base_w, base_h = base[2] - base[0], base[3] - base[1]
    last_risks: tuple[str, ...] = ()
    hard_safe_candidate: tuple[int, int, int, int] | None = None
    for factor in (1.0, 1.15, 1.35, 1.60):
        candidate = _centered_box(width, height, cx, cy, base_w * factor, base_h * factor)
        risks = _crop_risks(candidate, image_size, semantic_box, body)
        if not risks:
            hard_safe_candidate = candidate
            if _soft_joint_risk(candidate, image_size, body):
                continue
            saved_area = 1.0 - ((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])) / max(1, width * height)
            if saved_area < 0.08:
                return CropDecision(original, "original_fallback", "safe_original", ("crop_gain_too_small",))
            return CropDecision(candidate, f"body_aware_{base_strategy}", "safe", ())
        last_risks = risks
    if hard_safe_candidate is not None:
        saved_area = 1.0 - ((hard_safe_candidate[2] - hard_safe_candidate[0]) * (hard_safe_candidate[3] - hard_safe_candidate[1])) / max(1, width * height)
        if saved_area >= 0.08:
            return CropDecision(hard_safe_candidate, f"body_aware_{base_strategy}", "safe", ("noncritical_limb_crop",))
    return CropDecision(original, "original_fallback", "warning", last_risks or ("unsafe_crop",))


def _semantic_box(mask: Any, origin: tuple[int, int]) -> tuple[float, float, float, float] | None:
    import numpy as np

    useful = np.isin(mask, list(HEAD_LABELS))
    ys, xs = np.where(useful)
    if not len(xs):
        return None
    ox, oy = origin
    return float(xs.min() + ox), float(ys.min() + oy), float(xs.max() + 1 + ox), float(ys.max() + 1 + oy)


def _preview(image_bgr: Any, mask: Any) -> Any:
    import cv2
    import numpy as np

    colors = np.asarray(PALETTE, dtype=np.uint8)[mask]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mixed = np.where((mask > 0)[..., None], (0.55 * rgb + 0.45 * colors).astype(np.uint8), rgb)
    return mixed


def _body_preview(image_bgr: Any, body: Any | None, decision: CropDecision) -> Any:
    import cv2
    import numpy as np

    canvas = image_bgr.copy()
    if body is not None:
        contours, _ = cv2.findContours((body.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (60, 220, 60), max(2, round(min(canvas.shape[:2]) / 500)))
        landmarks = body.landmarks
        reliable = (landmarks[:, 3] >= 0.35) & (landmarks[:, 4] >= 0.60)
        for first, second in SKELETON:
            if reliable[first] and reliable[second]:
                cv2.line(canvas, tuple(landmarks[first, :2].astype(int)), tuple(landmarks[second, :2].astype(int)), (0, 210, 255), 3)
        for index in np.flatnonzero(reliable):
            cv2.circle(canvas, tuple(landmarks[index, :2].astype(int)), 4, (40, 40, 255), -1)
    color = (40, 210, 40) if decision.safety.startswith("safe") else (20, 170, 255)
    cv2.rectangle(canvas, decision.box[:2], decision.box[2:], color, max(3, round(min(canvas.shape[:2]) / 350)))
    label = f"{decision.safety}: {', '.join(decision.reasons) or 'body-aware crop'}"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 18 + len(label) * 10), 42), (0, 0, 0), -1)
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if max(canvas.shape[:2]) > 1280:
        scale = 1280 / max(canvas.shape[:2])
        canvas = cv2.resize(canvas, (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def prepare_dataset(
    root: Path,
    selected: list[Any],
    parser: Any,
    max_side: int = 1536,
    save_previews: bool = True,
    body_backend: Any | None = None,
    min_pose_confidence: float = 0.50,
    min_side: int = 512,
    *,
    skips: list[CropSkip] | None = None,
    rank_by_path: dict[Path, int] | None = None,
    crop_modes: dict[Path, str] | None = None,
    write_reports: bool = True,
) -> list[dict[str, object]]:
    import cv2
    from PIL import Image

    prepared_dir = root / "prepared"
    preview_dir = root / "parsing_preview"
    crop_preview_dir = root / "crop_preview"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    if save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
        crop_preview_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for position, record in enumerate(selected, 1):
        rank = position if rank_by_path is None else rank_by_path.get(record.path, position)

        def skip(reason: str) -> None:
            if skips is not None:
                skips.append(CropSkip(rank, record, reason))

        image = read_image(record.path)
        bbox_value = record.metadata.get("target_bbox")
        if image is None or not bbox_value:
            skip("unreadable_or_missing_target")
            continue
        height, width = image.shape[:2]
        bbox = tuple(map(float, bbox_value))
        px1, py1, px2, py2 = parsing_region(bbox, width, height)
        region = image[py1:py2, px1:px2]
        mask = parser.predict(region)
        semantic = _semantic_box(mask, (px1, py1))
        crop_mode = "standard" if crop_modes is None else crop_modes.get(record.path, "standard")
        body = None if body_backend is None or crop_mode == "tight" else body_backend.analyze(image, bbox)
        other_face_bboxes = [tuple(map(float, box)) for box in record.metadata.get("other_face_bboxes", [])]
        if crop_mode == "tight":
            tight_box, _ = recommended_crop((width, height), bbox, "close", semantic)
            decision = CropDecision(tight_box, "identity_tight", "safe", ("identity_only",))
        else:
            preparation_scale = "medium" if other_face_bboxes and record.scale_bin == "far" else record.scale_bin
            decision = body_aware_crop((width, height), bbox, preparation_scale, semantic, body, min_pose_confidence)
        safe_bounds = face_safe_bounds((width, height), bbox, other_face_bboxes, decision.box)
        if safe_bounds is None:
            skip("head_zones_overlap")
            continue
        if other_face_bboxes:
            safe_crop = _intersection(decision.box, safe_bounds)
            decision = CropDecision(
                safe_crop,
                f"{decision.strategy}_faces_excluded",
                decision.safety,
                (*decision.reasons, "other_faces_excluded"),
            )
        required_box = bbox
        if semantic is not None:
            required_box = (
                min(bbox[0], semantic[0]), min(bbox[1], semantic[1]),
                max(bbox[2], semantic[2]), max(bbox[3], semantic[3]),
            )
        crop_box, output_bucket = fit_crop_to_training_bucket(
            decision.box,
            (width, height),
            min_side,
            max_side,
            bounds=safe_bounds,
            required_box=required_box,
        )
        if output_bucket is None:
            skip("no_compatible_bucket")
            continue
        if any(_boxes_intersect(crop_box, other) for other in other_face_bboxes):
            skip("other_face_in_crop")
            continue
        if crop_box != decision.box:
            decision = CropDecision(crop_box, f"{decision.strategy}_bucket_expanded", decision.safety, decision.reasons)
        strategy = decision.strategy
        x1, y1, x2, y2 = crop_box
        crop_rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop_rgb)
        crop = crop.resize(output_bucket, Image.Resampling.LANCZOS)
        resolution_status = "ready"
        output_name = f"{rank:03d}_{record.path.stem}.jpg"
        crop.save(prepared_dir / output_name, quality=95, subsampling=0)
        if save_previews:
            Image.fromarray(_preview(region, mask)).save(preview_dir / f"{rank:03d}_{record.path.stem}.jpg", quality=92)
            Image.fromarray(_body_preview(image, body, decision)).save(crop_preview_dir / f"{rank:03d}_{record.path.stem}.jpg", quality=92)
        ratios = label_ratios(mask)
        manifest.append({
            "rank": rank,
            "source": str(record.path),
            "source_video": record.metadata.get("source_video", ""),
            "video_timestamp_seconds": record.metadata.get("video_timestamp_seconds", ""),
            "video_frame_number": record.metadata.get("video_frame_number", ""),
            "sface_similarity": "" if record.similarity is None else round(record.similarity, 5),
            "appearance_similarity": "" if record.appearance_similarity is None else round(record.appearance_similarity, 5),
            "prepared": str(prepared_dir / output_name),
            "strategy": strategy,
            "crop_safety": decision.safety,
            "crop_reasons": ";".join(decision.reasons),
            "crop_box": ",".join(map(str, crop_box)),
            "source_width": width,
            "source_height": height,
            "crop_width": x2 - x1,
            "crop_height": y2 - y1,
            "output_width": crop.width,
            "output_height": crop.height,
            "training_bucket": f"{output_bucket[0]}x{output_bucket[1]}",
            "resolution_status": resolution_status,
            "was_upscaled": False,
            "excluded_face_count": len(other_face_bboxes),
            "training_focus": "body" if record.body_focused else "identity",
            "skin_ratio": round(ratios.get("skin", 0.0), 5),
            "hair_ratio": round(ratios.get("hair", 0.0), 5),
            "glasses_ratio": round(ratios.get("glasses", 0.0), 5),
            "hat_ratio": round(ratios.get("hat", 0.0), 5),
            "body_found": body is not None,
            "pose_confidence": "" if body is None else round(body.confidence, 5),
            "body_detection_score": "" if body is None else round(body.detection_score, 5),
            "body_shape": record.body_shape or "",
            "chest_estimate": record.chest_estimate or "",
            "build": record.build or "",
            "shoulder_hip_ratio": "" if record.shoulder_hip_ratio is None else round(record.shoulder_hip_ratio, 5),
            "torso_proportion": "" if record.torso_proportion is None else round(record.torso_proportion, 5),
            "body_confidence": "" if record.body_confidence is None else round(record.body_confidence, 5),
        })
    if write_reports:
        write_preparation_reports(root, selected, manifest, save_previews)
    return manifest


def write_preparation_reports(
    root: Path, selected: list[Any | None], manifest: list[dict[str, object]], previews: bool,
) -> None:
    manifest_path = root / "dataset_manifest.csv"
    review_path = root / "review.html"
    if not manifest:
        manifest_path.unlink(missing_ok=True)
        review_path.unlink(missing_ok=True)
        return
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    _write_review_html(root, selected, manifest, previews)


def _write_review_html(root: Path, selected: list[Any | None], manifest: list[dict[str, object]], previews: bool) -> None:
    from urllib.parse import quote

    cards = []
    for row in manifest:
        rank = int(row["rank"])
        record = selected[rank - 1]
        if record is None:
            continue
        stem = record.path.stem
        original_name = f"{rank:03d}_{record.path.name}"
        prepared_name = Path(str(row["prepared"])).name
        preview_name = f"{rank:03d}_{stem}.jpg"
        images = [f'<figure><img src="selected/{quote(original_name)}"><figcaption>Оригінал</figcaption></figure>', f'<figure><img src="prepared/{quote(prepared_name)}"><figcaption>Рекомендований кроп</figcaption></figure>']
        if previews:
            images.append(f'<figure><img src="parsing_preview/{quote(preview_name)}"><figcaption>BiSeNet-маска</figcaption></figure>')
            images.append(f'<figure><img src="crop_preview/{quote(preview_name)}"><figcaption>Тіло, скелет і рішення</figcaption></figure>')
        similarity = "n/a" if record.similarity is None else f"{record.similarity:.3f}"
        appearance = "n/a" if record.appearance_similarity is None else f"{record.appearance_similarity:.3f}"
        body_desc = f" · body: {record.body_shape or '?'} · chest: {record.chest_estimate or '?'} · {record.build or '?'}" if record.body_shape else ""
        cards.append(
            '<article><h2>' + html.escape(f"#{rank} {record.path.name}") + '</h2>'
            + f'<p>SFace {similarity} · focus: <b>{html.escape(str(row["training_focus"]))}</b> · {html.escape(record.scale_bin)} · {html.escape(record.pose_bin)} · {html.escape(str(row["strategy"]))} · <b>{html.escape(str(row["crop_safety"]))}</b> {html.escape(str(row["crop_reasons"]))} · resolution: <b>{html.escape(str(row["resolution_status"]))}</b> ({row["output_width"]}×{row["output_height"]}){body_desc}</p>'
            + '<div class="images">' + "".join(images) + '</div></article>'
        )
    document = """<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>LoRA dataset review</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#151515;color:#eee}header{position:sticky;top:0;background:#151515e8;padding:8px 0;z-index:2}article{border-top:1px solid #444;padding:18px 0}h1,h2{margin:0 0 8px}p,figcaption{color:#bbb}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:340px;object-fit:contain;background:#080808;border-radius:6px}@media(max-width:1000px){.images{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.images{grid-template-columns:1fr}img{height:auto}}
</style><header><h1>Перевірка LoRA-датасету</h1><p>Оригінали не змінені. Для тренування використовуйте папку prepared/ після швидкого перегляду.</p></header>""" + "".join(cards) + "</html>"
    (root / "review.html").write_text(document, encoding="utf-8")
