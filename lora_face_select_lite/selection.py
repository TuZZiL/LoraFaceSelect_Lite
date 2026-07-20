from __future__ import annotations

from collections import Counter

from .metrics import hamming_distance
from .models import CandidateRecord


def _similarity(record: CandidateRecord) -> float:
    return -1.0 if record.similarity is None else record.similarity


def _base_score(record: CandidateRecord, minimum_similarity: float) -> float:
    similarity = _similarity(record)
    normalized = max(0.0, min(1.0, (similarity - minimum_similarity) / max(1e-6, 1.0 - minimum_similarity)))
    face_score = max(0.0, min(1.0, (record.face_width_ratio or 0.0) / 0.22))
    pose_score = 0.5 if record.yaw is None else max(0.0, 1.0 - abs(record.yaw) / 45.0)
    if record.appearance_similarity is not None:
        appearance = max(0.0, min(1.0, (record.appearance_similarity - 0.65) / 0.30))
        # SFace owns identity; MobileCLIP only anchors age/hair/makeup/look.
        return 0.30 * normalized + 0.40 * appearance + 0.12 * record.quality_score + 0.08 * face_score + 0.10 * pose_score
    # Preserve the lightweight behavior when the optional appearance model is absent.
    return 0.50 * normalized + 0.25 * record.quality_score + 0.15 * face_score + 0.10 * pose_score


def select_candidates(
    records: list[CandidateRecord],
    count: int,
    minimum_similarity: float,
    include_fallback: bool = False,
    duplicate_threshold: int = 6,
    diversity_strength: float = 0.22,
    initial_selected: list[CandidateRecord] | None = None,
    excluded_paths: set[str] | None = None,
) -> list[CandidateRecord]:
    if count <= 0:
        return []
    excluded_paths = excluded_paths or set()
    pool = [
        r for r in records
        if str(r.path) not in excluded_paths
        and r.status != "multiple_faces_review"
        and (r.eligible or (include_fallback and r.fallback_eligible))
    ]
    pool.sort(key=lambda r: (-_similarity(r), -r.quality_score, str(r.path).casefold(), str(r.path)))
    selected = list(initial_selected or [])
    selected_ids = {id(record) for record in selected}
    available_count = len({str(record.path) for record in [*pool, *selected]})
    body_available = (
        sum(record.body_focused for record in pool if id(record) not in selected_ids)
        + sum(record.body_focused for record in selected)
    )
    body_target = min(body_available, round(min(count, available_count) * 0.30))
    scale_counts: Counter[str] = Counter()
    pose_counts: Counter[str] = Counter()
    lighting_counts: Counter[str] = Counter()
    body_shape_counts: Counter[str] = Counter()
    build_counts: Counter[str] = Counter()
    body_selected = sum(record.body_focused for record in selected)
    for record in selected:
        scale_counts[record.scale_bin] += 1
        pose_counts[record.pose_bin] += 1
        lighting_counts[record.lighting_bin] += 1
        if record.body_shape:
            body_shape_counts[record.body_shape] += 1
        if record.build:
            build_counts[record.build] += 1

    def is_duplicate(candidate: CandidateRecord) -> bool:
        return any(hamming_distance(candidate.image_hash, item.image_hash) <= duplicate_threshold for item in selected)

    # Diversity is a soft reward. Never force a weak close/far image merely to
    # tick a scale bin; reference consistency has priority.
    while len(selected) < min(count, len(pool)):
        choices = [item for item in pool if item not in selected and not is_duplicate(item)]
        if not choices:
            break
        if body_selected < body_target:
            body_choices = [item for item in choices if item.body_focused]
            if body_choices:
                choices = body_choices

        def utility(item: CandidateRecord) -> float:
            # Give an unseen scale/pose a meaningful first-sample bonus. This
            # prevents a large medium/frontal burst from consuming the whole
            # set while still requiring the candidate to pass normal gates.
            scale_novelty = diversity_strength / (1 + scale_counts[item.scale_bin])
            pose_novelty = (diversity_strength * 0.45) / (1 + pose_counts[item.pose_bin])
            body_shape_novelty = 0.0
            build_novelty = 0.0
            if item.body_shape and item.body_shape != "unknown":
                body_shape_novelty = (diversity_strength * 0.15) / (1 + body_shape_counts[item.body_shape])
            if item.build and item.build != "unknown":
                build_novelty = (diversity_strength * 0.10) / (1 + build_counts[item.build])
            if item.appearance_similarity is not None:
                value = 0.90 * _base_score(item, minimum_similarity) + scale_novelty + pose_novelty + body_shape_novelty + build_novelty + 0.02 / (1 + lighting_counts[item.lighting_bin])
            else:
                value = 0.75 * _base_score(item, minimum_similarity) + scale_novelty + pose_novelty + body_shape_novelty + build_novelty + 0.05 / (1 + lighting_counts[item.lighting_bin])
            return value
        winner = max(choices, key=utility)
        selected.append(winner)
        scale_counts[winner.scale_bin] += 1
        pose_counts[winner.pose_bin] += 1
        lighting_counts[winner.lighting_bin] += 1
        if winner.body_shape:
            body_shape_counts[winner.body_shape] += 1
        if winner.build:
            build_counts[winner.build] += 1
        if winner.body_focused:
            body_selected += 1
    return selected
