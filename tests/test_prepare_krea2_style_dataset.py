from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image

import prepare_krea2_style_dataset as script
import prepare_krea2_style_dataset_gui as gui


def test_best_crop_plan_uses_closest_supported_bucket() -> None:
    plan = script.best_crop_plan(1200, 800, script.KREA2_BUCKETS)

    assert plan.bucket == (768, 512)
    assert plan.box == (0, 0, 1200, 800)


def test_best_crop_plan_avoids_upscale() -> None:
    plan = script.best_crop_plan(700, 1100, script.KREA2_BUCKETS)

    assert plan.bucket == (512, 768)


def test_crop_moves_to_keep_off_center_person() -> None:
    plan = script.centered_crop_plan(1600, 900, (1024, 768), ((1250, 100, 1550, 850),))

    assert plan.box[2] == 1600
    assert plan.box[0] > 0


def test_cli_converts_images_and_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    nested = source / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (1200, 800), "red").save(source / "wide.png")
    Image.new("RGBA", (800, 1000), (0, 0, 0, 0)).save(nested / "portrait.webp")

    result = subprocess.run(
        [sys.executable, str(Path(script.__file__)), str(source), str(output), "--no-smart-crop"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    outputs = sorted(output.rglob("*.jpg"))
    assert [path.relative_to(output).as_posix() for path in outputs] == ["nested/portrait.jpg", "wide.jpg"]
    with Image.open(output / "wide.jpg") as image:
        assert image.format == "JPEG"
        assert image.size == (768, 512)
    with Image.open(output / "nested" / "portrait.jpg") as image:
        assert image.mode == "RGB"
        assert image.size == (512, 768)
        assert image.getpixel((0, 0)) == (255, 255, 255)

    with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["bucket"] for row in rows} == {"768x512", "512x768"}


def test_gui_builds_cli_args_for_selected_options(tmp_path: Path) -> None:
    config = gui.RunConfig(
        input_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        models_dir=tmp_path / "models",
        quality=91,
        buckets=("512x768", "768x512"),
        smart_crop=False,
    )

    args = gui.build_cli_args(config)

    assert args[:2] == [sys.executable, "-u"]
    assert args[-5:] == ["--bucket", "512x768", "--bucket", "768x512", "--no-smart-crop"]
    assert args[args.index("--quality") + 1] == "91"


def test_gui_validation_rejects_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (output / "existing.jpg").touch()
    config = gui.RunConfig(source, output, tmp_path / "models", 95, ("512x512",), True)

    assert gui.validate_config(config) == f"Output folder must be empty:\n{output}"


def test_gui_builds_caption_arguments(tmp_path: Path) -> None:
    config = gui.RunConfig(
        input_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        models_dir=tmp_path / "models",
        quality=95,
        buckets=("512x512",),
        smart_crop=True,
        caption_prompt="Describe visible style only.",
        caption_max_tokens=96,
    )

    args = gui.build_cli_args(config)

    assert args[-4:] == ["--caption-prompt", "Describe visible style only.", "--caption-max-tokens", "96"]


def test_caption_prompt_is_editable_until_run_starts() -> None:
    class FakeControl:
        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def configure(self, **options: object) -> None:
            self.options.update(options)

    app = object.__new__(gui.KreaDatasetGUI)
    app.caption_prompt = FakeControl()
    app.caption_tokens = FakeControl()
    app._running = False

    app._sync_caption_controls()

    assert app.caption_prompt.options["state"] == "normal"
    assert app.caption_tokens.options["state"] == "normal"

    app._running = True
    app._sync_caption_controls()

    assert app.caption_prompt.options["state"] == "disabled"
    assert app.caption_tokens.options["state"] == "disabled"


def test_cli_writes_caption_sidecar_and_manifest(tmp_path: Path, monkeypatch: object) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    Image.new("RGB", (768, 768), "navy").save(source / "sample.jpg")

    class FakeCaptioner:
        def __init__(self, settings: object, log_path: Path) -> None:
            self.settings = settings
            self.log_path = log_path

        def start(self) -> None:
            pass

        def caption_image(self, image_path: Path) -> str:
            assert image_path.name == "sample.jpg"
            return "A square navy image with flat studio lighting."

        def close(self) -> None:
            pass

    monkeypatch.setattr(script, "LlamaCaptioner", FakeCaptioner)  # type: ignore[attr-defined]

    code = script.main(
        [
            str(source),
            str(output),
            "--no-smart-crop",
            "--caption-prompt",
            "Describe visible style only.",
        ]
    )

    assert code == 0
    assert (output / "sample.txt").read_text(encoding="utf-8") == (
        "A square navy image with flat studio lighting.\n"
    )
    with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["caption_status"] == "ok"
    assert row["caption"] == "A square navy image with flat studio lighting."


def test_cli_keeps_prepared_image_when_caption_server_fails(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    Image.new("RGB", (768, 768), "navy").save(source / "sample.jpg")

    class FailingCaptioner:
        def __init__(self, settings: object, log_path: Path) -> None:
            pass

        def start(self) -> None:
            raise script.CaptionError("model unavailable")

    monkeypatch.setattr(script, "LlamaCaptioner", FailingCaptioner)  # type: ignore[attr-defined]

    code = script.main(
        [
            str(source),
            str(output),
            "--no-smart-crop",
            "--caption-prompt",
            "Describe visible style only.",
        ]
    )

    assert code == 1
    assert (output / "sample.jpg").is_file()
    assert not (output / "sample.txt").exists()
    with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["caption_status"] == "error"
    assert row["caption_error"] == "model unavailable"
