from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from krea2_caption import (
    DEFAULT_CAPTION_MMPROJ,
    DEFAULT_CAPTION_MODEL,
    DEFAULT_LLAMA_SERVER,
    CaptionError,
    CaptionSettings,
    LlamaCaptioner,
)


INPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
KREA2_BUCKETS = tuple((width, height) for width in (512, 768, 1024) for height in (512, 768, 1024))


@dataclass(frozen=True)
class CropPlan:
    bucket: tuple[int, int]
    box: tuple[int, int, int, int]

    @property
    def kept_area(self) -> int:
        left, top, right, bottom = self.box
        return (right - left) * (bottom - top)


def parse_bucket(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid bucket '{value}'. Expected WIDTHxHEIGHT.") from exc
    if width not in {512, 768, 1024} or height not in {512, 768, 1024}:
        raise argparse.ArgumentTypeError("Each bucket side must be 512, 768, or 1024.")
    return width, height


def _focused_start(total: int, crop_size: int, focus_min: float, focus_max: float) -> int:
    if focus_max - focus_min <= crop_size:
        low = max(0, math.ceil(focus_max - crop_size))
        high = min(total - crop_size, math.floor(focus_min))
        return max(low, min(high, round((focus_min + focus_max - crop_size) / 2)))
    return max(0, min(total - crop_size, round((focus_min + focus_max - crop_size) / 2)))


def centered_crop_plan(
    width: int,
    height: int,
    bucket: tuple[int, int],
    focus_boxes: tuple[tuple[float, float, float, float], ...] = (),
) -> CropPlan:
    bucket_width, bucket_height = bucket
    if width * bucket_height > height * bucket_width:
        crop_height = height
        crop_width = math.floor(height * bucket_width / bucket_height)
    else:
        crop_width = width
        crop_height = math.floor(width * bucket_height / bucket_width)
    crop_width = max(1, crop_width)
    crop_height = max(1, crop_height)
    if focus_boxes:
        focus_left = min(box[0] for box in focus_boxes)
        focus_top = min(box[1] for box in focus_boxes)
        focus_right = max(box[2] for box in focus_boxes)
        focus_bottom = max(box[3] for box in focus_boxes)
        margin = min(width, height) * 0.025
        left = _focused_start(width, crop_width, max(0, focus_left - margin), min(width, focus_right + margin))
        top = _focused_start(height, crop_height, max(0, focus_top - margin), min(height, focus_bottom + margin))
    else:
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
    return CropPlan(bucket, (left, top, left + crop_width, top + crop_height))


def best_crop_plan(
    width: int,
    height: int,
    buckets: tuple[tuple[int, int], ...],
    focus_boxes: tuple[tuple[float, float, float, float], ...] = (),
) -> CropPlan:
    plans = [
        centered_crop_plan(width, height, bucket, focus_boxes)
        for bucket in buckets
        if bucket[0] <= width and bucket[1] <= height
    ]
    if not plans:
        raise ValueError("Image is smaller than every allowed bucket; upscaling is disabled.")
    source_area = width * height
    return max(plans, key=lambda plan: (plan.kept_area / source_area, plan.bucket[0] * plan.bucket[1]))


def flatten_transparency(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def detect_body_boxes(image: Image.Image, body_backend: object | None) -> tuple[tuple[float, float, float, float], ...]:
    if body_backend is None:
        return ()
    import cv2
    import numpy as np

    rgb = np.asarray(flatten_transparency(image))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    observations = body_backend.analyze_all(bgr)
    boxes = []
    for observation in observations:
        landmarks = observation.landmarks
        reliable = (landmarks[:, 3] >= 0.35) & (landmarks[:, 4] >= 0.60)
        points = landmarks[reliable, :2]
        if len(points):
            boxes.append(
                (
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                )
            )
        else:
            boxes.append(observation.bbox)
    return tuple(boxes)


def image_paths(root: Path, output: Path) -> list[Path]:
    output_resolved = output.resolve()
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in INPUT_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(output_resolved)
        except ValueError:
            paths.append(path)
    return sorted(paths, key=lambda path: str(path).casefold())


def prepare_image(
    source_path: Path,
    target_path: Path,
    buckets: tuple[tuple[int, int], ...],
    quality: int,
    body_backend: object | None,
) -> dict[str, object]:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source)
        original_width, original_height = image.size
        body_boxes = detect_body_boxes(image, body_backend)
        plan = best_crop_plan(original_width, original_height, buckets, body_boxes)
        target_size = plan.bucket
        crop = flatten_transparency(image.crop(plan.box))
        if crop.size != target_size:
            crop = crop.resize(target_size, Image.Resampling.LANCZOS)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target_path, "JPEG", quality=quality, subsampling=0, optimize=True)

    kept_percent = plan.kept_area / (original_width * original_height) * 100
    return {
        "source": str(source_path),
        "output": str(target_path),
        "original_size": f"{original_width}x{original_height}",
        "output_size": f"{target_size[0]}x{target_size[1]}",
        "bucket": f"{target_size[0]}x{target_size[1]}",
        "kept_percent": f"{kept_percent:.2f}",
        "cropped_percent": f"{100 - kept_percent:.2f}",
        "smart_crop": "pose" if body_boxes else "center",
        "people_detected": len(body_boxes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a folder of images to crop-minimized JPEGs for a Krea 2 style LoRA."
    )
    parser.add_argument("input", type=Path, help="Folder containing source images.")
    parser.add_argument("output", type=Path, help="New folder for prepared JPEGs.")
    parser.add_argument(
        "--bucket",
        action="append",
        dest="buckets",
        type=parse_bucket,
        metavar="WIDTHxHEIGHT",
        help="Allowed bucket. Repeat for multiple buckets. Each side must be 512, 768, or 1024.",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality from 1 to 100.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Folder containing MediaPipe ONNX models.")
    parser.add_argument("--no-smart-crop", action="store_true", help="Disable person/pose detection and center crops.")
    parser.add_argument("--caption-prompt", help="Generate a .txt caption for each prepared JPEG using this prompt.")
    parser.add_argument("--caption-max-tokens", type=int, default=160, help="Maximum caption response tokens.")
    parser.add_argument("--caption-server", type=Path, default=DEFAULT_LLAMA_SERVER, help="Path to llama-server executable.")
    parser.add_argument("--caption-model", type=Path, default=DEFAULT_CAPTION_MODEL, help="Path to caption model GGUF.")
    parser.add_argument("--caption-mmproj", type=Path, default=DEFAULT_CAPTION_MMPROJ, help="Path to vision projector GGUF.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    buckets = tuple(args.buckets or KREA2_BUCKETS)

    if not input_root.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output folder must be empty: {output_root}")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100.")
    if args.caption_prompt is not None and not args.caption_prompt.strip():
        raise SystemExit("--caption-prompt cannot be empty.")
    if not 1 <= args.caption_max_tokens <= 2048:
        raise SystemExit("--caption-max-tokens must be between 1 and 2048.")

    sources = image_paths(input_root, output_root)
    if not sources:
        raise SystemExit(f"No supported images found in: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    body_backend = None
    if not args.no_smart_crop:
        detector_path = args.models_dir / "person_detection_mediapipe_2023mar.onnx"
        pose_path = args.models_dir / "pose_estimation_mediapipe_2023mar.onnx"
        try:
            from lora_face_select_lite.body import MediaPipeBodyBackend

            body_backend = MediaPipeBodyBackend(detector_path, pose_path)
            print(f"Smart crop: {body_backend.provider}")
        except (ImportError, RuntimeError) as exc:
            print(f"Smart crop unavailable; using centered crops: {exc}", file=sys.stderr)
    captioner = None
    caption_start_error = ""
    if args.caption_prompt:
        settings = CaptionSettings(
            prompt=args.caption_prompt,
            server_path=args.caption_server.resolve(),
            model_path=args.caption_model.resolve(),
            mmproj_path=args.caption_mmproj.resolve(),
            max_tokens=args.caption_max_tokens,
        )
        captioner = LlamaCaptioner(settings, output_root / "caption_server.log")
        try:
            print("Caption model: loading Qwen3.5 vision model...")
            captioner.start()
            print("Caption model: ready")
        except CaptionError as exc:
            caption_start_error = str(exc)
            captioner = None
            print(f"Caption model unavailable; images will still be prepared: {exc}", file=sys.stderr)
    rows: list[dict[str, object]] = []
    failures: list[tuple[Path, str]] = []
    caption_failures: list[tuple[Path, str]] = []
    try:
        for index, source_path in enumerate(sources, start=1):
            relative = source_path.relative_to(input_root).with_suffix(".jpg")
            target_path = output_root / relative
            try:
                row = prepare_image(source_path, target_path, buckets, args.quality, body_backend)
                if args.caption_prompt:
                    if captioner is None:
                        row["caption_status"] = "error"
                        row["caption"] = ""
                        row["caption_error"] = caption_start_error
                        caption_failures.append((target_path, caption_start_error))
                    else:
                        try:
                            print(f"[{index}/{len(sources)}] CAPTION {relative}")
                            caption = captioner.caption_image(target_path)
                            target_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
                            row["caption_status"] = "ok"
                            row["caption"] = caption
                            row["caption_error"] = ""
                        except (CaptionError, OSError) as exc:
                            row["caption_status"] = "error"
                            row["caption"] = ""
                            row["caption_error"] = str(exc)
                            caption_failures.append((target_path, str(exc)))
                            print(f"[{index}/{len(sources)}] CAPTION ERROR {relative}: {exc}", file=sys.stderr)
                else:
                    row["caption_status"] = "disabled"
                    row["caption"] = ""
                    row["caption_error"] = ""
                rows.append(row)
                print(f"[{index}/{len(sources)}] {relative}")
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                failures.append((source_path, str(exc)))
                print(f"[{index}/{len(sources)}] SKIP {relative}: {exc}", file=sys.stderr)
    finally:
        if captioner is not None:
            captioner.close()

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "source", "output", "original_size", "output_size", "bucket",
            "kept_percent", "cropped_percent", "smart_crop", "people_detected",
            "caption_status", "caption", "caption_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Done: {len(rows)} JPEGs; skipped: {len(failures)}; "
        f"caption errors: {len(caption_failures)}; manifest: {manifest_path}"
    )
    return 1 if failures or caption_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
