from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import lora_face_select_lite.cli as cli
from lora_face_select_lite.models import CandidateRecord
from lora_face_select_lite.profiles import get_profile, model_manifest


def test_stable_profile_supports_legacy_model_layout(tmp_path: Path) -> None:
    profile = get_profile("stable")
    legacy = tmp_path / profile.models["face_detector"].filename
    legacy.write_bytes(b"legacy")
    assert profile.path(tmp_path, "face_detector") == legacy

    preferred = tmp_path / "stable" / legacy.name
    preferred.parent.mkdir()
    preferred.write_bytes(b"preferred")
    assert profile.path(tmp_path, "face_detector") == preferred


def test_experimental_profile_never_falls_back_to_stable_files(tmp_path: Path) -> None:
    profile = get_profile("experimental")
    legacy = tmp_path / profile.models["face_detector"].filename
    legacy.write_bytes(b"wrong location")
    assert profile.path(tmp_path, "face_detector") == tmp_path / "experimental" / legacy.name


def test_profile_manifest_records_precision_and_actual_checksum(tmp_path: Path) -> None:
    profile = get_profile("experimental")
    path = profile.path(tmp_path, "face_recognizer")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")
    manifest = model_manifest(profile, tmp_path, cli._file_sha256)
    assert manifest["face_detector"]["precision"] == "fp32"
    assert manifest["face_recognizer"]["architecture"] == "ArcFace-R50-w600k"
    assert manifest["face_recognizer"]["sha256"] == hashlib.sha256(b"model").hexdigest()
    assert manifest["appearance"]["available"] is False


def test_download_models_targets_only_requested_profile(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_download(url: str, destination: Path, checksum: str) -> None:
        calls.append((url, destination, checksum))

    monkeypatch.setattr(cli, "_download", fake_download)
    result = cli._run_models(SimpleNamespace(profile="experimental", models_dir=tmp_path, force=False))
    assert result == 0
    assert calls
    assert all(call[1].parent == tmp_path / "experimental" for call in calls)
    downloaded = {call[1].name for call in calls}
    assert downloaded >= {
        "face_detection_yunet_2023mar.onnx",
        "person_detection_mediapipe_2023mar.onnx",
        "pose_estimation_mediapipe_2023mar.onnx",
    }
    # The ArcFace R50 recognizer is a manual (URL-less) model and must not be
    # fetched automatically.
    assert "w600k_r50.onnx" not in downloaded


def _record(path: Path, *, similarity: float, faces: int = 1, status: str = "eligible") -> CandidateRecord:
    return CandidateRecord(path, status, similarity=similarity, face_count=faces, detection_score=0.9)


def test_benchmark_comparison_surfaces_selection_and_safety_regressions(tmp_path: Path) -> None:
    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    base_first = _record(first, similarity=0.8)
    base_second = _record(second, similarity=0.7)
    candidate_first = _record(first, similarity=0.79)
    candidate_second = _record(second, similarity=0.69, faces=0, status="rejected")
    base = {
        "profile": "stable",
        "records": [base_first, base_second],
        "selected": [base_first],
        "manifest": [{"source": str(first), "strategy": "smart", "crop_safety": "safe", "crop_box": "1,2,3,4"}],
        "analysis_seconds": 2.0,
        "preparation_seconds": 1.0,
    }
    candidate = {
        "profile": "experimental",
        "records": [candidate_first, candidate_second],
        "selected": [candidate_second],
        "manifest": [{"source": str(first), "strategy": "fallback", "crop_safety": "warning", "crop_box": "0,0,9,9"}],
        "analysis_seconds": 1.0,
        "preparation_seconds": 0.5,
    }

    rows, summary = cli._compare_benchmark_runs(base, candidate)

    assert summary["analysis_speedup"] == 2.0
    assert summary["new_face_misses"] == 1
    assert summary["unsafe_crop_regressions"] == 1
    assert summary["top_n_overlap_ratio"] == 0.0
    assert summary["acceptance_signals"]["analysis_speedup_at_least_15_percent"] is True
    assert sum(bool(row["selection_changed"]) for row in rows) == 2


def test_cli_exposes_profile_comparison_options() -> None:
    args = cli._build_parser().parse_args([
        "benchmark", "dataset", "--references", "reference.jpg",
        "--compare-profiles", "stable", "experimental",
    ])
    assert args.compare_profiles == ["stable", "experimental"]
    assert args.prepare_crops is True
