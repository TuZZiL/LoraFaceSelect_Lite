"""Compare MobileCLIP crop similarities with the existing SFace signal.

This is an intentionally optional experiment, not part of the application
dependencies. Run it from the dedicated .venv-mobileclip-win environment.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from lora_face_select_lite.backend import OpenCVYuNetSFaceBackend
from lora_face_select_lite.io_utils import image_paths, read_image
from lora_face_select_lite.metrics import cosine_similarity


def square_crop(image: Image.Image, bbox: tuple[float, float, float, float], factor: float, y_shift: float = 0.0) -> Image.Image:
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1) * factor
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 + y_shift * (y2 - y1)
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(image.width, int(round(cx + side / 2)))
    bottom = min(image.height, int(round(cy + side / 2)))
    return image.crop((left, top, right, bottom))


def embed(model, preprocess, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(image.convert("RGB")) for image in images[start : start + batch_size]])
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().numpy())
    return np.concatenate(chunks)


def load_model(architecture: str, checkpoint: Path):
    if architecture.startswith("MobileCLIP2"):
        try:
            import open_clip
        except ImportError as exc:
            raise SystemExit("MobileCLIP2 requires OpenCLIP with Apple's MobileCLIP2 patch.") from exc
        model, _, preprocess = open_clip.create_model_and_transforms(
            architecture,
            pretrained=str(checkpoint),
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
            device="cpu",
        )
        from mobileclip.modules.common.mobileone import reparameterize_model

        model = reparameterize_model(model)
    else:
        import mobileclip

        model, _, preprocess = mobileclip.create_model_and_transforms(architecture, pretrained=str(checkpoint), device="cpu")
    return model.eval(), preprocess


def parse_model(value: str) -> tuple[str, Path]:
    try:
        architecture, checkpoint = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use ARCHITECTURE=CHECKPOINT") from exc
    path = Path(checkpoint)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Checkpoint not found: {path}")
    return architecture, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("mobileclip_spike.csv"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        dest="model_specs",
        help="Repeat ARCHITECTURE=CHECKPOINT to compare models.",
    )
    args = parser.parse_args()

    cv2.setNumThreads(1)
    backend = OpenCVYuNetSFaceBackend(
        str(args.models / "face_detection_yunet_2023mar.onnx"),
        str(args.models / "face_recognition_sface_2021dec.onnx"),
    )
    reference_bgr = read_image(args.reference, max_side=960)
    if reference_bgr is None:
        raise SystemExit(f"Cannot read reference: {args.reference}")
    reference_faces = backend.analyze(reference_bgr)
    if len(reference_faces) != 1:
        raise SystemExit(f"Reference must have one face; found {len(reference_faces)}")
    reference_sface = reference_faces[0].embedding

    paths = image_paths(args.dataset)
    reference_original = ImageOps.exif_transpose(Image.open(args.reference)).convert("RGB")
    rsx = reference_original.width / reference_bgr.shape[1]
    rsy = reference_original.height / reference_bgr.shape[0]
    rx1, ry1, rx2, ry2 = reference_faces[0].bbox
    reference_bbox = rx1 * rsx, ry1 * rsy, rx2 * rsx, ry2 * rsy
    reference_crops = [
        square_crop(reference_original, reference_bbox, 1.20),
        square_crop(reference_original, reference_bbox, 1.90, -0.18),
        reference_original.copy(),
    ]
    rows, crops = [], []
    for path in paths:
        analysis = read_image(path, max_side=960)
        if analysis is None:
            continue
        faces = backend.analyze(analysis)
        if not faces:
            continue
        target = max(faces, key=lambda face: cosine_similarity(face.embedding, reference_sface))
        original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        sx, sy = original.width / analysis.shape[1], original.height / analysis.shape[0]
        x1, y1, x2, y2 = target.bbox
        bbox = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        # Face crop is tight; head crop includes hairstyle and some shoulders.
        crop_set = [square_crop(original, bbox, 1.20), square_crop(original, bbox, 1.90, -0.18), original.copy()]
        offset = len(crops)
        crops.extend(crop_set)
        rows.append({
            "path": str(path),
            "sface": cosine_similarity(target.embedding, reference_sface),
            "face_count": len(faces),
            "yaw": target.yaw,
            "crop_offset": offset,
        })

    specs = args.model_specs or [("mobileclip_s0", args.models / "mobileclip_s0.pt")]
    timings = []
    for architecture, checkpoint in specs:
        started = time.perf_counter()
        model, preprocess = load_model(architecture, checkpoint)
        features = embed(model, preprocess, [*reference_crops, *crops])
        elapsed = time.perf_counter() - started
        timings.append((architecture, elapsed))
        slug = architecture.lower().replace("-", "_")
        for row in rows:
            offset = int(row["crop_offset"]) + len(reference_crops)
            row[f"{slug}_face"] = float(features[offset] @ features[0])
            row[f"{slug}_head"] = float(features[offset + 1] @ features[1])
            row[f"{slug}_full"] = float(features[offset + 2] @ features[2])
    for row in rows:
        row.pop("crop_offset")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    timing_text = ", ".join(f"{name}={elapsed:.1f}s" for name, elapsed in timings)
    print(f"Embedded {len(rows)} images / {len(crops)} candidate crops; {timing_text}; wrote {args.output}")


if __name__ == "__main__":
    main()
