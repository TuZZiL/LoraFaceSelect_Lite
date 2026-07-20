from __future__ import annotations

import math
from typing import Any


def cosine_similarity(left: Any, right: Any) -> float:
    import numpy as np

    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return -1.0 if denominator == 0 else float(np.dot(a, b) / denominator)


def normalized_mean_embedding(embeddings: list[Any]) -> Any:
    import numpy as np

    if not embeddings:
        raise ValueError("At least one embedding is required")
    vectors = [np.asarray(item, dtype=np.float32).reshape(-1) for item in embeddings]
    dimensions = {vector.size for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("Reference embeddings must have the same size")
    norms = np.asarray([np.linalg.norm(vector) for vector in vectors], dtype=np.float32)
    if np.any(norms == 0):
        raise ValueError("Reference embeddings cannot be zero vectors")
    # Every reference contributes equally, regardless of the raw feature norm.
    vector = np.mean(np.asarray(vectors) / norms[:, None], axis=0)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Reference embeddings cannot average to zero")
    return vector / norm


def filter_outlier_indices(embeddings: list[Any], threshold: float = 0.60, min_keep: int = 2) -> list[int]:
    """Identify indices of outlier embeddings.

    Pairwise cosine similarity is used to find embeddings that are far from the
    consensus. Outlier removal is only performed if we have >= 3 embeddings,
    and we will never keep fewer than min_keep embeddings.
    """
    import numpy as np

    n = len(embeddings)
    if n < 3:
        return list(range(n))

    # Calculate pairwise similarity matrix
    sims = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sims[i, j] = 1.0
            else:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                sims[i, j] = sim
                sims[j, i] = sim

    # For each embedding, compute its mean similarity to all other embeddings
    mean_sims = []
    for i in range(n):
        other_sims = [sims[i, j] for j in range(n) if j != i]
        mean_sims.append(float(np.mean(other_sims)))

    # Median similarity of the consensus
    median_sim = float(np.median(mean_sims))

    # Sort indices by mean similarity (descending - best consensus first)
    sorted_indices = sorted(range(n), key=lambda idx: mean_sims[idx], reverse=True)

    # Identify non-outliers. We always keep the top `min_keep` regardless of score.
    keep_indices = set(sorted_indices[:min_keep])
    for idx in sorted_indices[min_keep:]:
        # If mean similarity to others is too low, we drop it
        if mean_sims[idx] >= threshold and mean_sims[idx] >= (median_sim - 0.15):
            keep_indices.add(idx)

    return sorted(list(keep_indices))


def quality_weighted_mean_embedding(embeddings: list[Any], qualities: list[float]) -> Any:
    """Compute a quality-weighted mean of embeddings and normalize the result."""
    import numpy as np

    if not embeddings:
        raise ValueError("At least one embedding is required")
    if len(embeddings) != len(qualities):
        raise ValueError("Embeddings and qualities lists must have the same length")

    vectors = [np.asarray(item, dtype=np.float32).reshape(-1) for item in embeddings]
    dimensions = {vector.size for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("Embeddings must have the same size")

    # Ensure all qualities are positive and non-zero
    weights = np.asarray([max(0.01, float(q)) for q in qualities], dtype=np.float32)
    weights /= weights.sum()

    # Compute quality-weighted mean
    vector = np.sum([v * w for v, w in zip(vectors, weights)], axis=0)
    norm = np.linalg.norm(vector)
    if norm == 0:
        # Fallback if somehow it cancels out
        vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Embeddings cannot average to zero")
    return vector / norm


def classify_scale(face_width_ratio: float) -> str:
    if face_width_ratio >= 0.22:
        return "close"
    if face_width_ratio >= 0.10:
        return "medium"
    return "far"


def classify_pose(yaw: float | None, pitch: float | None) -> str:
    if yaw is None:
        return "unknown"
    if yaw <= -40:
        return "left_profile"
    if yaw < -15:
        return "left_three_quarter"
    if yaw >= 40:
        return "right_profile"
    if yaw > 15:
        return "right_three_quarter"
    return "frontal"


def classify_lighting(mean_luma: float) -> str:
    if mean_luma < 0.30:
        return "dark"
    if mean_luma > 0.72:
        return "bright"
    return "normal"


def image_hash(image_bgr: Any) -> int:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] >= small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return int(value)


def hamming_distance(left: int | None, right: int | None) -> int:
    if left is None or right is None:
        return 64
    return (left ^ right).bit_count()


def image_quality(image_bgr: Any, bbox: tuple[float, float, float, float], resolution_scale: float = 1.0) -> dict[str, float]:
    import cv2
    import numpy as np

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return {"blur_raw": 0.0, "exposure_raw": 0.0, "resolution_raw": 0.0, "margin_raw": 0.0, "mean_luma": 0.5}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    luma = gray.astype(np.float32) / 255.0
    blur_raw = math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    clipped = float(((luma < 0.04) | (luma > 0.96)).mean())
    exposure_raw = max(0.0, 1.0 - clipped * 2.0) * max(0.0, 1.0 - abs(float(luma.mean()) - 0.5))
    resolution_raw = min(1.0, (x2 - x1) * max(1.0, resolution_scale) / 160.0)
    margin = min(x1, y1, width - x2, height - y2)
    margin_raw = min(1.0, max(0.0, margin / max(1.0, (x2 - x1) * 0.20)))
    return {"blur_raw": blur_raw, "exposure_raw": exposure_raw, "resolution_raw": resolution_raw, "margin_raw": margin_raw, "mean_luma": float(luma.mean())}
