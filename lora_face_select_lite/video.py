from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .metrics import cosine_similarity, hamming_distance, image_hash, image_quality

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    source_video: Path
    timestamp_seconds: float
    frame_number: int


@dataclass(frozen=True)
class VideoExtractionSummary:
    frames: list[ExtractedFrame]
    videos_found: int
    videos_processed: int
    videos_reused: int
    videos_failed: int
    sampled_frames: int


@dataclass
class _Candidate:
    image: Any
    timestamp_seconds: float
    frame_number: int
    score: float
    image_hash: int


ProgressCallback = Callable[[int, int, Path, str], None]


def video_paths(root: Path, recursive: bool = True) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: (str(path).casefold(), str(path)),
    )


def _identity_digest(identity: Any) -> str:
    try:
        return hashlib.sha256(identity.tobytes()).hexdigest()
    except AttributeError:
        return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()


def _config_signature(
    identity: Any,
    backend: Any,
    sample_fps: float,
    max_samples: int,
    max_candidates: int,
    analysis_size: int,
    min_similarity: float,
    min_face_width: int,
) -> str:
    values = {
        "identity": _identity_digest(identity),
        "backend": str(getattr(backend, "provider", type(backend).__name__)),
        "sample_fps": sample_fps,
        "max_samples": max_samples,
        "max_candidates": max_candidates,
        "analysis_size": analysis_size,
        "min_similarity": min_similarity,
        "min_face_width": min_face_width,
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "videos": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == 1 and isinstance(data.get("videos"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"version": 1, "videos": {}}


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _cached_frames(entry: Any, video: Path, signature: str) -> list[ExtractedFrame] | None:
    stat = video.stat()
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return None
    if entry.get("size") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns or entry.get("config") != signature:
        return None
    frames: list[ExtractedFrame] = []
    for item in entry.get("frames", []):
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            return None
        frames.append(ExtractedFrame(path, video, float(item["timestamp_seconds"]), int(item["frame_number"])))
    return frames


def _analysis_frame(image: Any, max_side: int) -> Any:
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _candidate_from_frame(
    image: Any,
    timestamp_seconds: float,
    frame_number: int,
    identity: Any,
    backend: Any,
    analysis_size: int,
    min_similarity: float,
    min_face_width: int,
) -> _Candidate | None:
    analysis = _analysis_frame(image, analysis_size)
    faces = backend.analyze(analysis)
    if not faces:
        return None
    similarity, target = max(
        ((cosine_similarity(face.embedding, identity), face) for face in faces),
        key=lambda item: item[0],
    )
    height, width = analysis.shape[:2]
    x1, _y1, x2, _y2 = target.bbox
    face_width = max(0.0, min(float(width), x2) - max(0.0, x1))
    if similarity < max(0.0, min_similarity - 0.15) or face_width < min_face_width or face_width / max(1, width) < 0.04:
        return None
    raw = image_quality(analysis, target.bbox)
    blur = max(0.0, min(1.0, raw["blur_raw"] / 8.0))
    exposure = max(0.0, min(1.0, raw["exposure_raw"]))
    resolution = max(0.0, min(1.0, raw["resolution_raw"]))
    margin = max(0.0, min(1.0, raw["margin_raw"]))
    quality = 0.40 * blur + 0.25 * exposure + 0.20 * resolution + 0.15 * margin
    score = 0.70 * similarity + 0.25 * quality + 0.05 * min(1.0, face_width / 160.0)
    return _Candidate(image, timestamp_seconds, frame_number, score, image_hash(analysis))


def _retain_candidate(candidates: list[_Candidate], candidate: _Candidate, limit: int) -> None:
    duplicate = next(
        (
            current
            for current in candidates
            if abs(current.timestamp_seconds - candidate.timestamp_seconds) < 1.5
            or hamming_distance(current.image_hash, candidate.image_hash) <= 6
        ),
        None,
    )
    if duplicate is not None:
        if candidate.score <= duplicate.score:
            return
        candidates.remove(duplicate)
    candidates.append(candidate)
    candidates.sort(key=lambda item: item.score, reverse=True)
    del candidates[limit:]


def _sample_positions(capture: Any, sample_fps: float, max_samples: int) -> list[tuple[float, int]]:
    import cv2

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        return []
    duration = frame_count / source_fps
    count = min(max_samples, max(1, math.ceil(duration * sample_fps)))
    step = duration / count
    timestamps = [min(duration, (index + 0.5) * step) for index in range(count)]
    return [(timestamp, min(frame_count - 1, round(timestamp * source_fps))) for timestamp in timestamps]


def _scan_video(
    video: Path,
    identity: Any,
    backend: Any,
    sample_fps: float,
    max_samples: int,
    max_candidates: int,
    analysis_size: int,
    min_similarity: float,
    min_face_width: int,
) -> tuple[list[_Candidate], int]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("cannot open video")
    candidates: list[_Candidate] = []
    sampled = 0
    try:
        positions = _sample_positions(capture, sample_fps, max_samples)
        if positions:
            for timestamp, frame_number in positions:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, image = capture.read()
                if not ok or image is None:
                    continue
                sampled += 1
                candidate = _candidate_from_frame(
                    image, timestamp, frame_number, identity, backend, analysis_size, min_similarity, min_face_width,
                )
                if candidate is not None:
                    _retain_candidate(candidates, candidate, max_candidates)
        else:
            # Corrupt or unusual containers may omit FPS/frame count. This
            # fallback still stays bounded and samples roughly every 30 frames.
            frame_number = 0
            while sampled < max_samples:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                if frame_number % 30 == 0:
                    timestamp = max(0.0, float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
                    sampled += 1
                    candidate = _candidate_from_frame(
                        image, timestamp, frame_number, identity, backend, analysis_size, min_similarity, min_face_width,
                    )
                    if candidate is not None:
                        _retain_candidate(candidates, candidate, max_candidates)
                frame_number += 1
    finally:
        capture.release()
    return candidates, sampled


def _write_candidates(directory: Path, video: Path, candidates: list[_Candidate]) -> list[ExtractedFrame]:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    video_id = hashlib.sha256(str(video.resolve()).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in video.stem)[:48] or "video"
    frames: list[ExtractedFrame] = []
    for candidate in sorted(candidates, key=lambda item: item.timestamp_seconds):
        timestamp_ms = round(candidate.timestamp_seconds * 1000)
        path = directory / f"{safe_stem}_{video_id}_{timestamp_ms:010d}.jpg"
        ok, encoded = cv2.imencode(".jpg", candidate.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"cannot encode frame at {candidate.timestamp_seconds:.3f}s")
        temporary = path.with_suffix(f"{path.suffix}.part")
        try:
            encoded.tofile(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        frames.append(ExtractedFrame(path, video, candidate.timestamp_seconds, candidate.frame_number))
    return frames


def extract_video_frames(
    videos: list[Path],
    output: Path,
    identity: Any,
    backend: Any,
    *,
    sample_fps: float = 0.5,
    max_samples: int = 120,
    max_candidates: int = 3,
    analysis_size: int = 720,
    min_similarity: float = 0.50,
    min_face_width: int = 48,
    progress: ProgressCallback | None = None,
) -> VideoExtractionSummary:
    frame_dir = output / "video_frames"
    checkpoint_path = output / "video_progress.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    signature = _config_signature(
        identity, backend, sample_fps, max_samples, max_candidates, analysis_size, min_similarity, min_face_width,
    )
    all_frames: list[ExtractedFrame] = []
    processed = reused = failed = sampled_total = 0
    for index, video in enumerate(videos, 1):
        key = str(video.resolve())
        cached = _cached_frames(checkpoint["videos"].get(key), video, signature)
        if cached is not None:
            reused += 1
            all_frames.extend(cached)
            if progress:
                progress(index, len(videos), video, f"reused {len(cached)}")
            continue
        try:
            candidates, sampled = _scan_video(
                video, identity, backend, sample_fps, max_samples, max_candidates,
                analysis_size, min_similarity, min_face_width,
            )
            frames = _write_candidates(frame_dir, video, candidates)
            stat = video.stat()
            checkpoint["videos"][key] = {
                "status": "ok",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "config": signature,
                "sampled_frames": sampled,
                "frames": [
                    {
                        "path": str(frame.path.resolve()),
                        "timestamp_seconds": round(frame.timestamp_seconds, 6),
                        "frame_number": frame.frame_number,
                    }
                    for frame in frames
                ],
            }
            processed += 1
            sampled_total += sampled
            all_frames.extend(frames)
            status = f"sampled {sampled}, kept {len(frames)}"
        except Exception as exc:
            checkpoint["videos"][key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            failed += 1
            status = f"ERROR: {exc}"
        _save_checkpoint(checkpoint_path, checkpoint)
        if progress:
            progress(index, len(videos), video, status)
    return VideoExtractionSummary(
        frames=all_frames,
        videos_found=len(videos),
        videos_processed=processed,
        videos_reused=reused,
        videos_failed=failed,
        sampled_frames=sampled_total,
    )
