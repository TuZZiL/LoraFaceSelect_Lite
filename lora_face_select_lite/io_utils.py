from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_dimensions(path: Path) -> tuple[int, int] | None:
    """Return EXIF-oriented (width, height) without decoding the full image."""
    from PIL import Image

    try:
        with Image.open(path) as source:
            width, height = source.size
            if source.getexif().get(274, 1) in {5, 6, 7, 8}:
                width, height = height, width
            return width, height
    except Exception:
        return None


def read_image(path: Path, max_side: int | None = None):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as source:
            if max_side:
                source.draft("RGB", (max_side, max_side))
            image = ImageOps.exif_transpose(source)
            if max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def image_paths(root: Path, recursive: bool = True) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: (str(p).casefold(), str(p)),
    )


def validate_output(root: Path, overwrite: bool = False) -> None:
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"Output path is not a folder: {root}")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"Output folder is not empty: {root}. Choose another folder or use --overwrite.")


def prepare_output(root: Path, overwrite: bool = False) -> None:
    validate_output(root, overwrite)
    if overwrite:
        for directory in (root / "selected", root / "multiple_faces_review", root / "crop_skipped", root / "prepared", root / "parsing_preview", root / "crop_preview"):
            if directory.exists():
                shutil.rmtree(directory)
        for filename in ("report.csv", "summary.json", "contact_sheet.jpg", "dataset_manifest.csv", "crop_skips.csv", "review.html"):
            path = root / filename
            if path.exists():
                path.unlink()
    (root / "selected").mkdir(parents=True, exist_ok=True)
    (root / "multiple_faces_review").mkdir(parents=True, exist_ok=True)


def _record_dict(record: object) -> dict[str, object]:
    from dataclasses import asdict
    data = asdict(record)
    data["path"] = str(data["path"])
    data.pop("metadata", None)
    return data


def write_reports(root: Path, records: list[object], selected: list[object], provider: str, requested: int, settings: dict[str, object]) -> None:
    selected_ids = {id(item) for item in selected}
    rows = []
    for record in records:
        row = _record_dict(record)
        metadata = getattr(record, "metadata", {})
        row["source_video"] = metadata.get("source_video", "")
        row["video_timestamp_seconds"] = metadata.get("video_timestamp_seconds", "")
        row["video_frame_number"] = metadata.get("video_frame_number", "")
        row["selected"] = id(record) in selected_ids
        if row["selected"]:
            row["decision_reason"] = "selected_strong" if row["status"] == "eligible" else "selected_fallback"
        elif row["status"] == "eligible":
            row["decision_reason"] = "not_selected"
        else:
            row["decision_reason"] = row["reason"]
        rows.append(row)
    fields = ["path", "source_video", "video_timestamp_seconds", "video_frame_number", "status", "reason", "decision_reason", "selected", "similarity", "appearance_similarity", "appearance_face_similarity", "appearance_head_similarity", "face_count", "face_width_ratio", "scale_bin", "pose_bin", "lighting_bin", "yaw", "pitch", "roll", "quality_score", "blur_score", "exposure_score", "resolution_score", "margin_score", "detection_score", "body_shape", "chest_estimate", "build", "shoulder_hip_ratio", "torso_proportion", "body_confidence", "nudenet_labels", "nudenet_max_score", "nudenet_error"]
    with (root / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "requested": requested,
        "selected": len(selected),
        "multiple_faces_review": sum(row["status"] == "multiple_faces_review" for row in rows),
        "rejected": sum(row["status"] == "rejected" for row in rows),
        "provider": provider,
        "settings": settings,
        "status_distribution": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
        "scale_distribution": _distribution(selected, "scale_bin"),
        "pose_distribution": _distribution(selected, "pose_bin"),
        "lighting_distribution": _distribution(selected, "lighting_bin"),
        "body_shape_distribution": _distribution(selected, "body_shape"),
        "chest_estimate_distribution": _distribution(selected, "chest_estimate"),
        "build_distribution": _distribution(selected, "build"),
        "records": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _distribution(records: list[object], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        value = str(getattr(record, field))
        result[value] = result.get(value, 0) + 1
    return result


def copy_results(root: Path, selected: list[object], review: list[object]) -> None:
    sync_selected(root, selected)
    used_names: set[str] = set()
    for record in review:
        destination = root / "multiple_faces_review" / record.path.name
        if destination.exists() or destination.name.casefold() in used_names:
            digest = hashlib.sha256(str(record.path.resolve()).encode("utf-8")).hexdigest()[:10]
            destination = root / "multiple_faces_review" / f"{record.path.stem}_{digest}{record.path.suffix}"
            suffix = 2
            while destination.exists() or destination.name.casefold() in used_names:
                destination = root / "multiple_faces_review" / f"{record.path.stem}_{digest}_{suffix}{record.path.suffix}"
                suffix += 1
        shutil.copy2(record.path, destination)
        used_names.add(destination.name.casefold())


def sync_selected(root: Path, ranked_records: list[object | None]) -> None:
    selected_dir = root / "selected"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True)
    for rank, record in enumerate(ranked_records, 1):
        if record is not None:
            shutil.copy2(record.path, selected_dir / f"{rank:03d}_{record.path.name}")


def write_crop_skips(root: Path, skips: list[object]) -> None:
    skipped_dir = root / "crop_skipped"
    report_path = root / "crop_skips.csv"
    if skipped_dir.exists():
        shutil.rmtree(skipped_dir)
    report_path.unlink(missing_ok=True)
    if not skips:
        return
    skipped_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for skip in sorted(skips, key=lambda item: item.rank):
        record = skip.record
        shutil.copy2(record.path, skipped_dir / f"{skip.rank:03d}_{record.path.name}")
        rows.append({
            "rank": skip.rank,
            "source": str(record.path),
            "reason": skip.reason,
            "similarity": "" if record.similarity is None else round(record.similarity, 5),
            "face_count": record.face_count,
            "scale_bin": record.scale_bin,
        })
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet_label(record: object, rank: int) -> str:
    score = "n/a" if getattr(record, "similarity", None) is None else f"{record.similarity:.3f}"
    chest = getattr(record, "chest_estimate", None) or "unknown"
    build = getattr(record, "build", None) or "unknown"
    return (
        f"{rank}: {record.scale_bin}, {record.pose_bin}\n"
        f"chest={chest}, build={build}\n"
        f"sim={score}"
    )


def make_contact_sheet(root: Path, selected: list[object], columns: int = 4) -> None:
    from PIL import Image, ImageDraw, ImageOps
    if not selected:
        sheet = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(sheet).text((24, 64), "No images selected", fill="black")
        sheet.save(root / "contact_sheet.jpg", quality=92)
        return
    thumb_w, thumb_h, label_h = 320, 260, 58
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(selected):
        try:
            with Image.open(record.path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((thumb_w - 12, thumb_h - 12))
            cell_x, cell_y = (index % columns) * thumb_w, (index // columns) * (thumb_h + label_h)
            sheet.paste(image, (cell_x + (thumb_w - image.width) // 2, cell_y + (thumb_h - image.height) // 2))
            draw.multiline_text(
                (cell_x + 6, cell_y + thumb_h),
                _contact_sheet_label(record, index + 1),
                fill="black",
                spacing=2,
            )
        except Exception:
            continue
    sheet.save(root / "contact_sheet.jpg", quality=92)
