from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen
from urllib.parse import quote

from .analysis import analyze_dataset, reference_appearance, reference_embedding
from .appearance import MobileCLIPAppearanceBackend
from .backend import OpenCVYuNetSFaceBackend
from .body import MediaPipeBodyBackend
from .io_utils import copy_results, image_paths, make_contact_sheet, prepare_output, read_image, sync_selected, validate_output, write_crop_skips, write_reports
from .metrics import cosine_similarity
from .nudenet import NudeNetBackend
from .parsing import BiSeNetFaceParser
from .preparation import CropSkip, prepare_dataset, write_preparation_reports
from .profiles import MODEL_PROFILES, MODEL_REVISION, get_profile, model_manifest
from .selection import select_candidates
from .video import ExtractedFrame, VideoExtractionSummary, extract_video_frames, video_paths

# Backwards-compatible public view used by older callers. New code should use
# MODEL_PROFILES so role, precision, licensing and optional models stay explicit.
_LEGACY_ROLE_NAMES = {
    "face_detector": "yunet",
    "face_recognizer": "sface",
    "face_parser": "bisenet",
    "person_detector": "persondet",
    "pose": "pose",
}
MODEL_SPECS = {
    _LEGACY_ROLE_NAMES[role]: {"filename": spec.filename, "url": spec.url, "sha256": spec.sha256}
    for role, spec in MODEL_PROFILES["stable"].models.items()
    if role in _LEGACY_ROLE_NAMES and spec.url is not None
}


@dataclass(frozen=True)
class CropShortfall:
    requested: int
    prepared: int
    skipped: int
    output: Path


CropDecisionCallback = Callable[[CropShortfall], str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    request = Request(url, headers={"User-Agent": "lora-face-select-lite/0.1"})
    try:
        digest = hashlib.sha256()
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual_sha256}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lora-face-select-lite", description="CPU-first face image selector for LoRA datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    models = sub.add_parser("download-models", help="Download face, parsing, person, pose, and NudeNet ONNX models.")
    models.add_argument("--models-dir", type=Path, default=Path("models"))
    models.add_argument("--profile", choices=(*MODEL_PROFILES, "all"), default="stable")
    models.add_argument("--force", action="store_true", help="Replace existing model files after verifying the downloads.")
    doctor = sub.add_parser("doctor", help="Check OpenCV and model setup.")
    doctor.add_argument("--models-dir", type=Path, default=Path("models"))
    doctor.add_argument("--model-profile", choices=MODEL_PROFILES, default="stable")
    doctor.add_argument("--opencv-threads", type=int, default=1)
    benchmark = sub.add_parser("benchmark", help="Measure detection/recognition without copying files.")
    benchmark.add_argument("dataset", type=Path)
    benchmark.add_argument("--references", nargs="+", required=True, metavar="PATH")
    benchmark.add_argument("--models-dir", type=Path, default=Path("models"))
    benchmark.add_argument("--model-profile", choices=MODEL_PROFILES, default="stable")
    benchmark.add_argument("--compare-profiles", nargs=2, choices=MODEL_PROFILES, metavar=("BASE", "CANDIDATE"))
    benchmark.add_argument("--output", type=Path, default=Path("benchmark_report"))
    benchmark.add_argument("--overwrite", action="store_true")
    benchmark.add_argument("--count", type=int, default=20)
    benchmark.add_argument("--min-similarity", type=float, default=0.50)
    benchmark.add_argument("--duplicate-hamming-threshold", type=int, default=6)
    benchmark.add_argument("--diversity-strength", type=float, default=0.22)
    benchmark.add_argument("--min-face-width", type=int, default=48)
    benchmark.add_argument("--min-quality", type=float, default=0.15)
    benchmark.add_argument("--max-abs-yaw", type=float, default=25.0)
    benchmark.add_argument("--prepare-crops", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--crop-min-side", type=int, choices=(512, 768, 1024), default=512)
    benchmark.add_argument("--crop-max-side", type=int, choices=(512, 768, 1024), default=1024)
    benchmark.add_argument("--min-body-pose-confidence", type=float, default=0.50)
    benchmark.add_argument("--analysis-size", type=int, default=960)
    benchmark.add_argument("--opencv-threads", type=int, default=1)
    benchmark.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--body-attributes", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--nudenet-model", type=Path, default=None)
    select = sub.add_parser("select", help="Analyze a dataset and copy selected images.")
    select.add_argument("dataset", type=Path)
    select.add_argument("--references", nargs="+", required=True, metavar="PATH")
    select.add_argument("--count", type=int, default=20)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--models-dir", type=Path, default=Path("models"))
    select.add_argument("--model-profile", choices=MODEL_PROFILES, default="stable")
    select.add_argument("--min-similarity", type=float, default=0.50)
    select.add_argument("--duplicate-hamming-threshold", type=int, default=6, help="Maximum pHash Hamming distance for near-duplicate suppression.")
    select.add_argument("--diversity-strength", type=float, default=0.22, help="First-sample bonus for unseen scale/pose bins (0 disables diversity bonuses).")
    select.add_argument("--analysis-size", type=int, default=960)
    select.add_argument("--opencv-threads", type=int, default=1)
    select.add_argument("--min-face-width", type=int, default=48, help="Minimum detected face width at the analysis resolution.")
    select.add_argument("--min-quality", type=float, default=0.15, help="Reject candidates below this combined quality score (0..1).")
    select.add_argument("--max-abs-yaw", type=float, default=25.0, help="Maximum coarse left/right face angle; 0 disables this filter.")
    select.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    select.add_argument("--overwrite", action="store_true")
    select.add_argument("--shortfall", choices=("prompt", "strict", "fill"), default="prompt")
    select.add_argument("--prepare-crops", action=argparse.BooleanOptionalAction, default=True, help="Create LoRA-ready smart crops using BiSeNet face parsing.")
    select.add_argument("--crop-min-side", type=int, choices=(512, 768, 1024), default=512, help="Smallest allowed Krea 2 training-bucket side; images are never upscaled.")
    select.add_argument("--crop-max-side", type=int, choices=(512, 768, 1024), default=1024, help="Largest allowed Krea 2 training-bucket side; images are never upscaled.")
    select.add_argument("--parsing-previews", action=argparse.BooleanOptionalAction, default=True, help="Save face/hair parsing previews for quick review.")
    select.add_argument("--min-body-pose-confidence", type=float, default=0.50, help="Use an automatic crop only above this body pose confidence (0..1).")
    select.add_argument("--appearance-rerank", action=argparse.BooleanOptionalAction, default=True, help="Use optional MobileCLIP ONNX to anchor age, hair, makeup, and overall look when the local model is available.")
    select.add_argument("--appearance-model", type=Path, default=None, help="Path to image-only MobileCLIP-S0 ONNX (defaults to models/mobileclip_s0_image.onnx).")
    select.add_argument("--body-attributes", action=argparse.BooleanOptionalAction, default=True, help="Estimate body attributes from MediaPipe pose and annotate target-person body-part detections with NudeNet when available.")
    select.add_argument("--nudenet-model", type=Path, default=None, help="Path to NudeNet v3 320n ONNX (defaults to the selected model profile).")
    select.add_argument("--analyze-videos", action=argparse.BooleanOptionalAction, default=True, help="Extract target-person candidates from local videos found in the dataset.")
    select.add_argument("--video-sample-fps", type=float, default=0.5, help="How many video frames per second to inspect.")
    select.add_argument("--video-max-samples", type=int, default=120, help="Maximum inspected frames per video.")
    select.add_argument("--video-max-candidates", type=int, default=3, help="Maximum diverse frames retained from each video.")
    return parser


def _reference_files(values: list[str]) -> list[Path]:
    result: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            result.extend(image_paths(path))
        elif path.is_file():
            result.append(path)
        else:
            raise ValueError(f"Reference path does not exist: {path}")
    result = sorted(set(result), key=lambda p: (str(p).casefold(), str(p)))
    if not 1 <= len(result) <= 5:
        raise ValueError("Provide between 1 and 5 reference images.")
    return result


def _backend(models_dir: Path, opencv_threads: int = 1, model_profile: str = "stable") -> OpenCVYuNetSFaceBackend:
    if not 1 <= opencv_threads <= 64:
        raise ValueError("--opencv-threads must be between 1 and 64")
    import cv2

    cv2.setNumThreads(opencv_threads)
    profile = get_profile(model_profile)
    detector = profile.models["face_detector"]
    recognizer = profile.models["face_recognizer"]
    return OpenCVYuNetSFaceBackend(
        str(profile.path(models_dir, "face_detector")),
        str(profile.path(models_dir, "face_recognizer")),
        profile=profile.name,
        precision=f"{detector.precision}+{recognizer.precision}",
        recognizer_input_name=recognizer.input_name,
        recognizer_kind=recognizer.recognizer_kind,
    )


def _profile_parser(models_dir: Path, model_profile: str) -> BiSeNetFaceParser:
    profile = get_profile(model_profile)
    return BiSeNetFaceParser(profile.path(models_dir, "face_parser"))


def _profile_body(models_dir: Path, model_profile: str) -> MediaPipeBodyBackend:
    profile = get_profile(model_profile)
    return MediaPipeBodyBackend(
        profile.path(models_dir, "person_detector"),
        profile.path(models_dir, "pose"),
    )


def _profile_appearance(models_dir: Path, model_profile: str, override: Path | None = None) -> tuple[Path, str]:
    profile = get_profile(model_profile)
    spec = profile.models["appearance"]
    return (override or profile.path(models_dir, "appearance"), spec.architecture if override is None else override.stem)


def _profile_nudenet(models_dir: Path, model_profile: str, override: Path | None = None) -> tuple[Path, str]:
    profile = get_profile(model_profile)
    spec = profile.models["nudenet"]
    return (override or profile.path(models_dir, "nudenet"), spec.architecture if override is None else override.stem)


def _run_models(args: argparse.Namespace) -> int:
    profile_names = list(MODEL_PROFILES) if args.profile == "all" else [args.profile]
    for profile_name in profile_names:
        profile = get_profile(profile_name)
        for role, spec in profile.models.items():
            if spec.url is None or spec.sha256 is None:
                print(f"manual: {profile_name}/{role} -> {profile.directory(args.models_dir) / spec.filename} ({spec.license})")
                continue
            destination = profile.download_path(args.models_dir, role)
            if destination.exists():
                actual_sha256 = _file_sha256(destination)
                if actual_sha256 == spec.sha256:
                    print(f"verified: {destination}")
                    continue
                if not args.force:
                    raise RuntimeError(f"Existing model has an unexpected checksum: {destination}. Use --force to replace it.")
            print(f"downloading {profile_name}/{role} -> {destination}")
            _download(spec.url, destination, spec.sha256)
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is missing. Install with: python -m pip install -e '.[test]'") from exc
    print(f"opencv: {cv2.__version__}")
    profile = get_profile(args.model_profile)
    backend = _backend(args.models_dir, args.opencv_threads, profile.name)
    print(f"model_profile: {profile.name} — {profile.description}")
    print(f"provider: {backend.provider}")
    print(f"opencv_threads: {cv2.getNumThreads()}")
    print("face detector and recognizer initialized successfully")
    parser = _profile_parser(args.models_dir, profile.name)
    print(f"face_parser: {parser.provider}")
    body = _profile_body(args.models_dir, profile.name)
    print(f"body_analyzer: {body.provider}")
    appearance_path, architecture = _profile_appearance(args.models_dir, profile.name)
    if appearance_path.is_file():
        appearance = MobileCLIPAppearanceBackend(appearance_path, architecture=architecture)
        print(f"optional_appearance: {appearance.provider}")
    else:
        print("optional_appearance: unavailable (MobileCLIP ONNX not installed)")
    nudenet_path, _ = _profile_nudenet(args.models_dir, profile.name)
    if nudenet_path.is_file():
        try:
            nudenet = NudeNetBackend(nudenet_path)
            print(f"optional_nudenet: {nudenet.provider}")
        except (RuntimeError, ValueError) as exc:
            print(f"optional_nudenet: unavailable (failed to load {nudenet_path}: {exc})", file=sys.stderr)
    else:
        print("optional_nudenet: unavailable (NudeNet ONNX not installed)")
    return 0


def _run_select(args: argparse.Namespace, crop_decision: CropDecisionCallback | None = None) -> int:
    if not args.dataset.is_dir():
        raise ValueError(f"Dataset folder does not exist: {args.dataset}")
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    if not 0.0 <= args.min_similarity <= 1.0:
        raise ValueError("--min-similarity must be between 0 and 1")
    if args.duplicate_hamming_threshold < 0:
        raise ValueError("--duplicate-hamming-threshold must be non-negative")
    if args.diversity_strength < 0.0:
        raise ValueError("--diversity-strength must be non-negative")
    if args.analysis_size < 320:
        raise ValueError("--analysis-size must be at least 320")
    if args.min_face_width <= 0:
        raise ValueError("--min-face-width must be greater than zero")
    if not 0.0 <= args.min_quality <= 1.0:
        raise ValueError("--min-quality must be between 0 and 1")
    if not 0.0 <= args.max_abs_yaw <= 90.0:
        raise ValueError("--max-abs-yaw must be between 0 and 90")
    if args.crop_min_side <= 0:
        raise ValueError("--crop-min-side must be greater than zero")
    if args.crop_max_side < args.crop_min_side:
        raise ValueError("--crop-max-side must be at least --crop-min-side")
    if not 0.0 <= args.min_body_pose_confidence <= 1.0:
        raise ValueError("--min-body-pose-confidence must be between 0 and 1")
    if not 0.0 < args.video_sample_fps <= 10.0:
        raise ValueError("--video-sample-fps must be greater than 0 and at most 10")
    if not 1 <= args.video_max_samples <= 10000:
        raise ValueError("--video-max-samples must be between 1 and 10000")
    if not 1 <= args.video_max_candidates <= 100:
        raise ValueError("--video-max-candidates must be between 1 and 100")
    references = _reference_files(args.references)
    output_resolved = args.output.resolve()
    reference_set = {path.resolve() for path in references}
    if any(path.is_relative_to(output_resolved) for path in reference_set):
        raise ValueError("Reference images must be outside the output folder to protect them from --overwrite.")
    validate_output(args.output, args.overwrite)
    paths = [p for p in image_paths(args.dataset, args.recursive) if p.resolve() not in reference_set and not p.resolve().is_relative_to(output_resolved)]
    videos = [p for p in video_paths(args.dataset, args.recursive) if not p.resolve().is_relative_to(output_resolved)] if args.analyze_videos else []
    if not paths and not videos:
        raise ValueError("No supported images or videos found in the dataset.")
    profile = get_profile(args.model_profile)
    backend = _backend(args.models_dir, args.opencv_threads, profile.name)
    identity = reference_embedding(references, backend, args.analysis_size)
    appearance_backend = None
    appearance_identity = None
    if args.appearance_rerank:
        appearance_path, architecture = _profile_appearance(args.models_dir, profile.name, args.appearance_model)
        if appearance_path.is_file():
            appearance_backend = MobileCLIPAppearanceBackend(appearance_path, architecture=architecture)
            appearance_identity = reference_appearance(references, backend, appearance_backend, args.analysis_size)
            print(f"Appearance reranking enabled: {appearance_backend.provider}")
        else:
            print(f"Warning: optional appearance model not found: {appearance_path}; continuing with SFace only.", file=sys.stderr)
    body_backend = None
    nudenet_backend = None
    if args.body_attributes:
        body_backend = _profile_body(args.models_dir, profile.name)
        nudenet_path, _ = _profile_nudenet(args.models_dir, profile.name, args.nudenet_model)
        if nudenet_path.is_file():
            try:
                nudenet_backend = NudeNetBackend(nudenet_path)
                print(f"Body attributes enabled: {body_backend.provider} + {nudenet_backend.provider}")
            except (RuntimeError, ValueError) as exc:
                nudenet_backend = None
                print(f"Body attributes enabled: landmarks only (NudeNet ONNX failed to load: {exc})", file=sys.stderr)
        else:
            print("Body attributes enabled: landmarks only (NudeNet ONNX not installed).")
    started = time.perf_counter()
    output_prepared = False
    video_summary: VideoExtractionSummary | None = None
    video_metadata: dict[Path, ExtractedFrame] = {}
    if videos:
        prepare_output(args.output, args.overwrite)
        output_prepared = True
        print(f"Scanning {len(videos)} local videos sequentially...")
        video_summary = extract_video_frames(
            videos,
            args.output,
            identity,
            backend,
            sample_fps=args.video_sample_fps,
            max_samples=args.video_max_samples,
            max_candidates=args.video_max_candidates,
            analysis_size=min(args.analysis_size, 720),
            min_similarity=args.min_similarity,
            min_face_width=args.min_face_width,
            progress=lambda current, total, path, status: print(f"{current}/{total}: {path.name[:60]} — {status}"),
        )
        paths.extend(frame.path for frame in video_summary.frames)
        video_metadata = {frame.path.resolve(): frame for frame in video_summary.frames}
        print(
            f"Video frames: {len(video_summary.frames)} kept; "
            f"{video_summary.videos_processed} processed, {video_summary.videos_reused} reused, "
            f"{video_summary.videos_failed} failed."
        )
    if not paths:
        raise ValueError("No usable image candidates were found in the dataset videos.")
    print(f"Analyzing {len(paths)} images with {backend.provider}...")
    records = analyze_dataset(
        paths,
        identity,
        backend,
        args.min_similarity,
        min_face_width=args.min_face_width,
        min_quality=args.min_quality,
        max_abs_yaw=None if args.max_abs_yaw == 0 else args.max_abs_yaw,
        max_side=args.analysis_size,
        progress=lambda current, total, path: print(f"\r{current}/{total}: {path.name[:70]:<70}", end="", flush=True),
        appearance_identity=appearance_identity,
        appearance_backend=appearance_backend,
        body_backend=body_backend,
        nudenet_backend=nudenet_backend,
        min_body_pose_confidence=args.min_body_pose_confidence,
    )
    print()
    for record in records:
        frame = video_metadata.get(record.path.resolve())
        if frame is not None:
            record.metadata.update({
                "source_video": str(frame.source_video.resolve()),
                "video_timestamp_seconds": round(frame.timestamp_seconds, 6),
                "video_frame_number": frame.frame_number,
            })
    selected = select_candidates(records, args.count, args.min_similarity, duplicate_threshold=args.duplicate_hamming_threshold, diversity_strength=args.diversity_strength)
    if len(selected) < args.count:
        fallback = args.shortfall == "fill"
        if args.shortfall == "prompt" and sys.stdin.isatty():
            print(f"Only {len(selected)} strong candidates found for requested {args.count}.")
            fallback = input("Fill with lower-similarity single-face images? [y/N] ").strip().lower() in {"y", "yes"}
        if fallback:
            selected = select_candidates(records, args.count, args.min_similarity, include_fallback=True, duplicate_threshold=args.duplicate_hamming_threshold, diversity_strength=args.diversity_strength)
        if len(selected) < args.count:
            print(f"Warning: selecting {len(selected)} images instead of {args.count}.", file=sys.stderr)
    if not output_prepared:
        prepare_output(args.output, args.overwrite)
    copy_results(args.output, selected, [r for r in records if r.status == "multiple_faces_review"])
    selected_slots: list[object | None] = [*selected, *([None] * max(0, args.count - len(selected)))]
    final_selected = list(selected)
    crop_skips: list[CropSkip] = []
    prepared_rows: list[dict[str, object]] = []
    preparation_safety: dict[str, int] = {}
    crop_resolution_action = "none"
    base_prepared_count = 0
    if args.prepare_crops:
        print("Preparing smart crops with BiSeNet face parsing...")
        parser = _profile_parser(args.models_dir, profile.name)
        body = _profile_body(args.models_dir, profile.name)
        prepared_rows = prepare_dataset(
            args.output,
            selected,
            parser,
            args.crop_max_side,
            args.parsing_previews,
            body,
            args.min_body_pose_confidence,
            args.crop_min_side,
            skips=crop_skips,
        )
        for skip in crop_skips:
            selected_slots[skip.rank - 1] = None
        write_crop_skips(args.output, crop_skips)
        base_prepared_count = len(prepared_rows)

        action = "finish"
        if crop_skips and crop_decision is not None:
            action = crop_decision(CropShortfall(args.count, len(prepared_rows), len(crop_skips), args.output))
        crop_resolution_action = action if crop_skips else "none"
        if action == "tight":
            retry_skips: list[CropSkip] = []
            retry_records = [skip.record for skip in crop_skips]
            rank_by_path = {skip.record.path: skip.rank for skip in crop_skips}
            retry_rows = prepare_dataset(
                args.output,
                retry_records,
                parser,
                args.crop_max_side,
                args.parsing_previews,
                body,
                args.min_body_pose_confidence,
                args.crop_min_side,
                skips=retry_skips,
                rank_by_path=rank_by_path,
                crop_modes={record.path: "tight" for record in retry_records},
                write_reports=False,
            )
            retry_successes = {str(row["source"]) for row in retry_rows}
            for skip in crop_skips:
                if str(skip.record.path) in retry_successes:
                    selected_slots[skip.rank - 1] = skip.record
            prepared_rows.extend(retry_rows)
            crop_skips = retry_skips
        elif action == "backfill":
            excluded_paths = {str(skip.record.path) for skip in crop_skips}
            unresolved_skips = list(crop_skips)
            while any(record is None for record in selected_slots):
                current = [record for record in selected_slots if record is not None]
                expanded = select_candidates(
                    records,
                    args.count,
                    args.min_similarity,
                    include_fallback=args.shortfall == "fill",
                    duplicate_threshold=args.duplicate_hamming_threshold,
                    diversity_strength=args.diversity_strength,
                    initial_selected=current,
                    excluded_paths=excluded_paths,
                )
                newcomers = expanded[len(current):]
                holes = [index for index, record in enumerate(selected_slots) if record is None]
                if not newcomers:
                    break
                attempts = newcomers[:len(holes)]
                attempt_ranks = {record.path: holes[index] + 1 for index, record in enumerate(attempts)}
                attempt_skips: list[CropSkip] = []
                attempt_rows = prepare_dataset(
                    args.output,
                    attempts,
                    parser,
                    args.crop_max_side,
                    args.parsing_previews,
                    body,
                    args.min_body_pose_confidence,
                    args.crop_min_side,
                    skips=attempt_skips,
                    rank_by_path=attempt_ranks,
                    write_reports=False,
                )
                successful_paths = {str(row["source"]) for row in attempt_rows}
                for record in attempts:
                    if str(record.path) in successful_paths:
                        selected_slots[attempt_ranks[record.path] - 1] = record
                    else:
                        excluded_paths.add(str(record.path))
                prepared_rows.extend(attempt_rows)
                unresolved_skips.extend(attempt_skips)
            crop_skips = unresolved_skips

        prepared_rows.sort(key=lambda row: int(row["rank"]))
        final_selected = [record for record in selected_slots if record is not None]
        sync_selected(args.output, selected_slots)
        write_crop_skips(args.output, crop_skips)
        write_preparation_reports(args.output, selected_slots, prepared_rows, args.parsing_previews)
        preparation_safety = dict(sorted(Counter(str(row["crop_safety"]) for row in prepared_rows).items()))
    prepared_count = len(prepared_rows)
    elapsed = time.perf_counter() - started
    if not args.prepare_crops:
        session_status = "completed"
    elif prepared_count < args.count:
        session_status = "completed_with_skips"
    elif crop_resolution_action == "backfill":
        session_status = "completed_with_replacements"
    elif crop_resolution_action == "tight":
        session_status = "completed_after_tighter_crop"
    else:
        session_status = "completed"
    write_reports(args.output, records, final_selected, backend.provider, args.count, {"dataset": str(args.dataset.resolve()), "references": [str(p.resolve()) for p in references], "count": args.count, "model_profile": profile.name, "models": model_manifest(profile, args.models_dir, _file_sha256), "min_similarity": args.min_similarity, "profile_recommended_min_similarity": profile.recommended_min_similarity, "min_face_width": args.min_face_width, "min_quality": args.min_quality, "max_abs_yaw": args.max_abs_yaw, "analysis_size": args.analysis_size, "opencv_threads": args.opencv_threads, "shortfall": args.shortfall, "appearance_rerank": appearance_backend is not None, "appearance_provider": None if appearance_backend is None else appearance_backend.provider, "prepare_crops": args.prepare_crops, "crop_max_side": args.crop_max_side, "parsing_previews": args.parsing_previews, "min_body_pose_confidence": args.min_body_pose_confidence, "analyze_videos": args.analyze_videos, "video_sample_fps": args.video_sample_fps, "video_max_samples": args.video_max_samples, "video_max_candidates": args.video_max_candidates, "videos_found": 0 if video_summary is None else video_summary.videos_found, "videos_processed": 0 if video_summary is None else video_summary.videos_processed, "videos_reused": 0 if video_summary is None else video_summary.videos_reused, "videos_failed": 0 if video_summary is None else video_summary.videos_failed, "video_frames_kept": 0 if video_summary is None else len(video_summary.frames), "video_frames_sampled_this_run": 0 if video_summary is None else video_summary.sampled_frames, "prepared_count": prepared_count, "base_prepared_count": base_prepared_count, "resolved_crop_count": max(0, prepared_count - base_prepared_count), "crop_skipped_count": len(crop_skips), "crop_resolution_action": crop_resolution_action, "session_status": session_status, "preparation_safety": preparation_safety, "elapsed_seconds": round(elapsed, 3)})
    make_contact_sheet(args.output, final_selected)
    print(f"Selected {len(final_selected)} images -> {args.output / 'selected'}")
    if args.prepare_crops:
        print(f"Prepared {prepared_count} smart crops -> {args.output / 'prepared'}")
        if crop_skips:
            print(f"Skipped {len(crop_skips)} crops -> {args.output / 'crop_skipped'}")
        print(f"Crop safety: {preparation_safety}")
    print(f"Report -> {args.output / 'report.csv'}")
    return 0


def _benchmark_profile(
    args: argparse.Namespace,
    profile_name: str,
    references: list[Path],
    paths: list[Path],
    root: Path,
) -> dict[str, object]:
    profile = get_profile(profile_name)
    backend = _backend(args.models_dir, args.opencv_threads, profile_name)
    identity = reference_embedding(references, backend, args.analysis_size)
    started = time.perf_counter()
    records = analyze_dataset(
        paths,
        identity,
        backend,
        args.min_similarity,
        min_face_width=args.min_face_width,
        min_quality=args.min_quality,
        max_abs_yaw=None if args.max_abs_yaw == 0 else args.max_abs_yaw,
        max_side=args.analysis_size,
    )
    analysis_seconds = time.perf_counter() - started
    selected = select_candidates(records, args.count, args.min_similarity)
    prepare_output(root)
    copy_results(root, selected, [])
    manifest: list[dict[str, object]] = []
    preparation_seconds = 0.0
    if args.prepare_crops:
        parser = _profile_parser(args.models_dir, profile_name)
        body = _profile_body(args.models_dir, profile_name)
        started = time.perf_counter()
        manifest = prepare_dataset(
            root,
            selected,
            parser,
            args.crop_max_side,
            False,
            body,
            args.min_body_pose_confidence,
            args.crop_min_side,
        )
        preparation_seconds = time.perf_counter() - started
    settings = {
        "model_profile": profile_name,
        "models": model_manifest(profile, args.models_dir, _file_sha256),
        "min_similarity": args.min_similarity,
        "analysis_size": args.analysis_size,
        "analysis_seconds": round(analysis_seconds, 6),
        "preparation_seconds": round(preparation_seconds, 6),
    }
    write_reports(root, records, selected, backend.provider, args.count, settings)
    return {
        "profile": profile_name,
        "records": records,
        "selected": selected,
        "manifest": manifest,
        "analysis_seconds": analysis_seconds,
        "preparation_seconds": preparation_seconds,
        "provider": backend.provider,
        "models": settings["models"],
    }


def _optional_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _compare_benchmark_runs(base: dict[str, object], candidate: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    base_records = {str(record.path.resolve()): record for record in base["records"]}  # type: ignore[index, union-attr]
    candidate_records = {str(record.path.resolve()): record for record in candidate["records"]}  # type: ignore[index, union-attr]
    base_ranks = {str(record.path.resolve()): rank for rank, record in enumerate(base["selected"], 1)}  # type: ignore[index, union-attr]
    candidate_ranks = {str(record.path.resolve()): rank for rank, record in enumerate(candidate["selected"], 1)}  # type: ignore[index, union-attr]
    base_crops = {str(Path(str(row["source"])).resolve()): row for row in base["manifest"]}  # type: ignore[index]
    candidate_crops = {str(Path(str(row["source"])).resolve()): row for row in candidate["manifest"]}  # type: ignore[index]
    rows: list[dict[str, object]] = []
    deltas: list[float] = []
    for path in sorted(base_records.keys() | candidate_records.keys(), key=lambda value: value.casefold()):
        left, right = base_records.get(path), candidate_records.get(path)
        left_similarity = None if left is None else _optional_number(left.similarity)
        right_similarity = None if right is None else _optional_number(right.similarity)
        delta = None if left_similarity is None or right_similarity is None else right_similarity - left_similarity
        if delta is not None:
            deltas.append(delta)
        left_crop, right_crop = base_crops.get(path), candidate_crops.get(path)
        left_rank, right_rank = base_ranks.get(path), candidate_ranks.get(path)
        row = {
            "path": path,
            "base_face_count": "" if left is None else left.face_count,
            "candidate_face_count": "" if right is None else right.face_count,
            "face_count_changed": left is None or right is None or left.face_count != right.face_count,
            "base_similarity": "" if left_similarity is None else round(left_similarity, 6),
            "candidate_similarity": "" if right_similarity is None else round(right_similarity, 6),
            "similarity_delta": "" if delta is None else round(delta, 6),
            "base_detection_score": "" if left is None or left.detection_score is None else round(left.detection_score, 6),
            "candidate_detection_score": "" if right is None or right.detection_score is None else round(right.detection_score, 6),
            "base_status": "missing" if left is None else left.status,
            "candidate_status": "missing" if right is None else right.status,
            "status_changed": left is None or right is None or left.status != right.status or left.reason != right.reason,
            "base_reason": "missing" if left is None else left.reason,
            "candidate_reason": "missing" if right is None else right.reason,
            "base_selected_rank": "" if left_rank is None else left_rank,
            "candidate_selected_rank": "" if right_rank is None else right_rank,
            "selection_changed": (left_rank is None) != (right_rank is None),
            "base_crop_strategy": "" if left_crop is None else left_crop["strategy"],
            "candidate_crop_strategy": "" if right_crop is None else right_crop["strategy"],
            "base_crop_safety": "" if left_crop is None else left_crop["crop_safety"],
            "candidate_crop_safety": "" if right_crop is None else right_crop["crop_safety"],
            "base_crop_box": "" if left_crop is None else left_crop["crop_box"],
            "candidate_crop_box": "" if right_crop is None else right_crop["crop_box"],
            "crop_changed": left_crop is not None and right_crop is not None and (
                left_crop["strategy"] != right_crop["strategy"]
                or left_crop["crop_safety"] != right_crop["crop_safety"]
                or left_crop["crop_box"] != right_crop["crop_box"]
            ),
        }
        rows.append(row)
    base_selected = set(base_ranks)
    candidate_selected = set(candidate_ranks)
    overlap = len(base_selected & candidate_selected)
    overlap_ratio = overlap / max(1, len(base_selected))
    base_seconds = float(base["analysis_seconds"])
    candidate_seconds = float(candidate["analysis_seconds"])
    speedup = base_seconds / candidate_seconds if candidate_seconds > 0 else None
    new_face_misses = sum(
        left.face_count > 0 and candidate_records.get(path) is not None and candidate_records[path].face_count == 0
        for path, left in base_records.items()
    )
    unsafe_regressions = sum(
        str(row["base_crop_safety"]).startswith("safe")
        and row["candidate_crop_safety"] != ""
        and not str(row["candidate_crop_safety"]).startswith("safe")
        for row in rows
    )
    summary = {
        "base_profile": base["profile"],
        "candidate_profile": candidate["profile"],
        "image_count": len(rows),
        "base_analysis_seconds": round(base_seconds, 6),
        "candidate_analysis_seconds": round(candidate_seconds, 6),
        "analysis_speedup": None if speedup is None else round(speedup, 4),
        "base_preparation_seconds": round(float(base["preparation_seconds"]), 6),
        "candidate_preparation_seconds": round(float(candidate["preparation_seconds"]), 6),
        "similarity_pairs": len(deltas),
        "mean_similarity_delta": None if not deltas else round(sum(deltas) / len(deltas), 6),
        "median_similarity_delta": None if not deltas else round(statistics.median(deltas), 6),
        "max_abs_similarity_delta": None if not deltas else round(max(abs(value) for value in deltas), 6),
        "status_changes": sum(bool(row["status_changed"]) for row in rows),
        "face_count_changes": sum(bool(row["face_count_changed"]) for row in rows),
        "new_face_misses": new_face_misses,
        "selection_changes": sum(bool(row["selection_changed"]) for row in rows),
        "top_n_overlap": overlap,
        "top_n_overlap_ratio": round(overlap_ratio, 4),
        "crop_changes": sum(bool(row["crop_changed"]) for row in rows),
        "unsafe_crop_regressions": unsafe_regressions,
        "acceptance_signals": {
            "no_new_face_misses": new_face_misses == 0,
            "no_unsafe_crop_regressions": unsafe_regressions == 0,
            "top_n_overlap_at_least_90_percent": overlap_ratio >= 0.90,
            "analysis_speedup_at_least_15_percent": speedup is not None and speedup >= 1.15,
        },
        "manual_review_required": True,
    }
    return rows, summary


def _write_comparison_html(root: Path, rows: list[dict[str, object]], title: str, field: str, profiles: tuple[str, str]) -> None:
    cards = []
    for row in rows:
        if not row[field]:
            continue
        source = Path(str(row["path"]))
        source_url = quote(source.as_posix())
        details = html.escape(
            f"{profiles[0]}: {row['base_status']} sim={row['base_similarity']} rank={row['base_selected_rank']} "
            f"| {profiles[1]}: {row['candidate_status']} sim={row['candidate_similarity']} rank={row['candidate_selected_rank']}"
        )
        images = f'<figure><img src="{source_url}"><figcaption>Source</figcaption></figure>'
        if field == "crop_changed":
            for prefix, profile in (("base", profiles[0]), ("candidate", profiles[1])):
                rank = row[f"{prefix}_selected_rank"]
                if rank != "":
                    crop_name = f"{int(rank):03d}_{source.stem}.jpg"
                    crop_url = quote(f"profiles/{profile}/prepared/{crop_name}")
                    images += f'<figure><img src="{crop_url}"><figcaption>{html.escape(profile)} crop</figcaption></figure>'
        cards.append(f"<article><h2>{html.escape(source.name)}</h2><p>{details}</p><div>{images}</div></article>")
    document = """<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<style>body{font-family:system-ui;background:#151515;color:#eee;margin:24px}article{border-top:1px solid #444;padding:16px 0}p,figcaption{color:#bbb}article>div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:360px;object-fit:contain;background:#080808}</style>"""
    document += f"<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>" + "".join(cards) + "</html>"
    (root / ("changed_crops.html" if field == "crop_changed" else "changed_selections.html")).write_text(document, encoding="utf-8")


def _run_comparative_benchmark(args: argparse.Namespace, references: list[Path], paths: list[Path]) -> int:
    base_name, candidate_name = args.compare_profiles
    if base_name == candidate_name:
        raise ValueError("--compare-profiles requires two different profiles")
    validate_output(args.output, args.overwrite)
    if args.overwrite:
        profiles_root = args.output / "profiles"
        if profiles_root.exists():
            shutil.rmtree(profiles_root)
        for filename in ("summary.json", "comparison.csv", "changed_selections.html", "changed_crops.html"):
            path = args.output / filename
            if path.exists():
                path.unlink()
    args.output.mkdir(parents=True, exist_ok=True)
    profiles_root = args.output / "profiles"
    print(f"Benchmarking profile: {base_name}")
    base = _benchmark_profile(args, base_name, references, paths, profiles_root / base_name)
    print(f"Benchmarking profile: {candidate_name}")
    candidate = _benchmark_profile(args, candidate_name, references, paths, profiles_root / candidate_name)
    rows, summary = _compare_benchmark_runs(base, candidate)
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["settings"] = {
        "dataset": str(args.dataset.resolve()),
        "references": [str(path.resolve()) for path in references],
        "count": args.count,
        "min_similarity": args.min_similarity,
        "min_face_width": args.min_face_width,
        "min_quality": args.min_quality,
        "max_abs_yaw": args.max_abs_yaw,
        "analysis_size": args.analysis_size,
        "prepare_crops": args.prepare_crops,
        "crop_min_side": args.crop_min_side,
        "crop_max_side": args.crop_max_side,
        "min_body_pose_confidence": args.min_body_pose_confidence,
    }
    median_delta = summary.get("median_similarity_delta")
    summary["threshold_calibration"] = {
        "base_min_similarity": args.min_similarity,
        "suggested_candidate_start": (
            args.min_similarity if median_delta is None
            else round(max(0.0, min(1.0, args.min_similarity + float(median_delta))), 4)
        ),
        "method": "base threshold plus median paired-score delta; validate manually near the gate",
    }
    summary["models"] = {base_name: base["models"], candidate_name: candidate["models"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_comparison_html(args.output, rows, "Changed selections", "selection_changed", (base_name, candidate_name))
    _write_comparison_html(args.output, rows, "Changed crops", "crop_changed", (base_name, candidate_name))
    print(f"Comparison report -> {args.output}")
    print(json.dumps(summary["acceptance_signals"], ensure_ascii=False))
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    if not args.dataset.is_dir():
        raise ValueError(f"Dataset folder does not exist: {args.dataset}")
    if args.analysis_size < 320:
        raise ValueError("--analysis-size must be at least 320")
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    if not 0.0 <= args.min_similarity <= 1.0:
        raise ValueError("--min-similarity must be between 0 and 1")
    if args.crop_min_side <= 0:
        raise ValueError("--crop-min-side must be greater than zero")
    if args.crop_max_side < args.crop_min_side:
        raise ValueError("--crop-max-side must be at least --crop-min-side")
    references = _reference_files(args.references)
    reference_set = {path.resolve() for path in references}
    output_resolved = args.output.resolve()
    paths = [
        path for path in image_paths(args.dataset, args.recursive)
        if path.resolve() not in reference_set and not path.resolve().is_relative_to(output_resolved)
    ]
    if not paths:
        raise ValueError("No supported images found in the dataset.")
    if args.compare_profiles:
        return _run_comparative_benchmark(args, references, paths)
    backend = _backend(args.models_dir, args.opencv_threads, args.model_profile)
    identity = reference_embedding(references, backend, args.analysis_size)
    print("path | face_count | best_similarity | detection_score | elapsed_ms")
    for path in paths:
        started = time.perf_counter()
        image = read_image(path, max_side=args.analysis_size)
        faces = [] if image is None else backend.analyze(image)
        if faces:
            score, target = max(((cosine_similarity(face.embedding, identity), face) for face in faces), key=lambda item: item[0])
            detection = f"{target.detection_score:.4f}"
            similarity = f"{score:.4f}"
        else:
            detection = similarity = "n/a"
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"{path} | {len(faces)} | {similarity} | {detection} | {elapsed_ms:.1f}")
    return 0


def main(argv: list[str] | None = None, crop_decision: CropDecisionCallback | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "download-models":
            return _run_models(args)
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "benchmark":
            return _run_benchmark(args)
        return _run_select(args, crop_decision)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
