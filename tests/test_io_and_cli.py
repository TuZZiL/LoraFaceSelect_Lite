from __future__ import annotations

import hashlib
import io
import json
import queue
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import lora_face_select_lite.cli as cli
from lora_face_select_lite.gui import LoraFaceSelectGUI, _Redirect
from lora_face_select_lite.io_utils import _contact_sheet_label, copy_results, image_dimensions, make_contact_sheet, prepare_output, read_image, sync_selected, write_crop_skips, write_reports
from lora_face_select_lite.models import CandidateRecord


def test_gui_redirect_emits_carriage_return_progress_immediately() -> None:
    messages: "queue.Queue[str]" = queue.Queue()
    redirect = _Redirect(messages)

    redirect.write("\r12/1710: photo.jpg")

    assert messages.get_nowait() == "12/1710: photo.jpg\n"


def test_gui_passes_video_and_resume_settings_to_cli() -> None:
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    gui = object.__new__(LoraFaceSelectGUI)
    values = {
        "dataset_var": "dataset", "references_var": "one.jpg two.jpg", "output_var": "result",
        "models_dir_var": "models", "model_profile_var": "experimental", "count_var": "20",
        "min_similarity_var": "0.50",
        "min_quality_var": "0.15", "max_abs_yaw_var": "25", "min_face_width_var": "48",
        "video_sample_fps_var": "0.25", "video_max_samples_var": "80",
        "video_max_candidates_var": "4", "prepare_crops_var": True,
        "appearance_rerank_var": True, "body_attributes_var": True,
        "parsing_previews_var": True, "analyze_videos_var": True, "overwrite_var": True,
    }
    for name, value in values.items():
        setattr(gui, name, Value(value))

    argv = gui._select_argv()

    assert argv[argv.index("--model-profile") + 1] == "experimental"
    assert argv[argv.index("--video-sample-fps") + 1] == "0.25"
    assert argv[argv.index("--video-max-samples") + 1] == "80"
    assert argv[argv.index("--video-max-candidates") + 1] == "4"
    assert "--overwrite" in argv


def test_contact_sheet_is_created_for_empty_selection(tmp_path: Path) -> None:
    prepare_output(tmp_path)
    make_contact_sheet(tmp_path, [])
    with Image.open(tmp_path / "contact_sheet.jpg") as image:
        assert image.size == (640, 160)


def test_contact_sheet_label_includes_chest_and_build(tmp_path: Path) -> None:
    record = CandidateRecord(
        tmp_path / "person.jpg",
        "eligible",
        similarity=0.8123,
        scale_bin="medium",
        pose_bin="frontal",
        chest_estimate="medium",
        build="athletic",
    )
    assert _contact_sheet_label(record, 2) == (
        "2: medium, frontal\n"
        "chest=medium, build=athletic\n"
        "sim=0.812"
    )


def test_contact_sheet_label_marks_missing_body_estimates_as_unknown(tmp_path: Path) -> None:
    record = CandidateRecord(tmp_path / "person.jpg", "eligible", scale_bin="close", pose_bin="profile")
    label = _contact_sheet_label(record, 1)
    assert "chest=unknown, build=unknown" in label


def test_read_image_honors_exif_unicode_and_never_changes_source(tmp_path: Path) -> None:
    source = tmp_path / "обличчя.jpg"
    image = Image.new("RGB", (8, 4), (120, 80, 40))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    before = source.read_bytes()

    loaded = read_image(source)

    assert loaded is not None
    assert loaded.shape[:2] == (8, 4)
    assert image_dimensions(source) == (4, 8)
    assert source.read_bytes() == before


def test_review_collision_names_are_stable(tmp_path: Path) -> None:
    left = tmp_path / "left" / "same.jpg"
    right = tmp_path / "right" / "same.jpg"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"left")
    right.write_bytes(b"right")

    names_by_run = []
    for name in ("result_one", "result_two"):
        output = tmp_path / name
        prepare_output(output)
        copy_results(output, [], [SimpleNamespace(path=left), SimpleNamespace(path=right)])
        names_by_run.append(sorted(path.name for path in (output / "multiple_faces_review").iterdir()))

    assert names_by_run[0] == names_by_run[1]
    assert names_by_run[0][0] == "same.jpg"
    assert len(names_by_run[0]) == 2


def test_selected_and_crop_skipped_outputs_follow_final_slots(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    skipped = tmp_path / "skipped.jpg"
    replacement = tmp_path / "replacement.jpg"
    for path in (first, skipped, replacement):
        path.write_bytes(path.stem.encode())
    first_record = CandidateRecord(first, "eligible", similarity=.9)
    skipped_record = CandidateRecord(skipped, "eligible", similarity=.8, face_count=2)
    replacement_record = CandidateRecord(replacement, "eligible", similarity=.7)
    prepare_output(tmp_path / "result")

    sync_selected(tmp_path / "result", [first_record, replacement_record])
    write_crop_skips(
        tmp_path / "result",
        [SimpleNamespace(rank=2, record=skipped_record, reason="no_compatible_bucket")],
    )

    assert sorted(path.name for path in (tmp_path / "result" / "selected").iterdir()) == ["001_first.jpg", "002_replacement.jpg"]
    assert [path.name for path in (tmp_path / "result" / "crop_skipped").iterdir()] == ["002_skipped.jpg"]
    assert "no_compatible_bucket" in (tmp_path / "result" / "crop_skips.csv").read_text(encoding="utf-8")


def test_reports_explain_selected_and_unselected_candidates(tmp_path: Path) -> None:
    prepare_output(tmp_path)
    selected = CandidateRecord(Path("selected.jpg"), "eligible", similarity=0.9)
    omitted = CandidateRecord(Path("omitted.jpg"), "eligible", similarity=0.8)
    rejected = CandidateRecord(Path("rejected.jpg"), "rejected", reason="no_face")

    write_reports(tmp_path, [selected, omitted, rejected], [selected], "test", 1, {"min_similarity": 0.35})
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    reasons = {Path(row["path"]).name: row["decision_reason"] for row in summary["records"]}

    assert reasons == {
        "selected.jpg": "selected_strong",
        "omitted.jpg": "not_selected",
        "rejected.jpg": "no_face",
    }
    assert summary["status_distribution"] == {"eligible": 2, "rejected": 1}


def test_overwrite_does_not_touch_output_before_backend_validation(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    reference = tmp_path / "reference.jpg"
    candidate = dataset / "candidate.jpg"
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    def fail_backend(*args, **kwargs):
        raise RuntimeError("invalid models")

    monkeypatch.setattr(cli, "_backend", fail_backend)
    result = cli.main([
        "select",
        str(dataset),
        "--references",
        str(reference),
        "--output",
        str(output),
        "--overwrite",
    ])

    assert result == 2
    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_reference_inside_output_is_rejected_before_overwrite(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "candidate.jpg").write_bytes(b"candidate")
    output = tmp_path / "output"
    reference = output / "selected" / "reference.jpg"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    monkeypatch.setattr(cli, "_backend", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend must not start")))

    result = cli.main([
        "select",
        str(dataset),
        "--references",
        str(reference),
        "--output",
        str(output),
        "--overwrite",
    ])

    assert result == 2
    assert reference.read_bytes() == b"reference"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_is_atomic_and_checksum_verified(monkeypatch, tmp_path: Path) -> None:
    payload = b"model data"
    destination = tmp_path / "model.onnx"
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _Response(payload))

    cli._download("https://example.invalid/model", destination, hashlib.sha256(payload).hexdigest())
    assert destination.read_bytes() == payload

    bad_destination = tmp_path / "bad.onnx"
    try:
        cli._download("https://example.invalid/model", bad_destination, "0" * 64)
    except RuntimeError as exc:
        assert "Checksum mismatch" in str(exc)
    else:
        raise AssertionError("checksum mismatch was not rejected")
    assert not bad_destination.exists()
    assert not (tmp_path / "bad.onnx.part").exists()
