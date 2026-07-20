from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"


@dataclass(frozen=True)
class ModelSpec:
    role: str
    filename: str
    url: str | None
    sha256: str | None
    precision: str
    architecture: str
    required: bool = True
    license: str = ""
    input_name: str | None = None
    # Recognizer preprocessing family; "sface" (OpenCV FaceRecognizerSF) or
    # "arcface" (InsightFace 5-point alignment + OpenCV DNN). Only meaningful
    # for the face_recognizer role.
    recognizer_kind: str = "sface"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    description: str
    models: dict[str, ModelSpec]
    recommended_min_similarity: float
    legacy_fallback: bool = False

    def directory(self, models_dir: Path) -> Path:
        return models_dir / self.name

    def path(self, models_dir: Path, role: str) -> Path:
        spec = self.models[role]
        preferred = self.directory(models_dir) / spec.filename
        if preferred.is_file() or not self.legacy_fallback:
            return preferred
        legacy = models_dir / spec.filename
        return legacy if legacy.is_file() else preferred

    def download_path(self, models_dir: Path, role: str) -> Path:
        spec = self.models[role]
        if self.legacy_fallback:
            legacy = models_dir / spec.filename
            if legacy.is_file():
                return legacy
        return self.directory(models_dir) / spec.filename


def _zoo_url(path: str) -> str:
    return f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_REVISION}/models/{path}"


_NUDENET_URL = "https://raw.githubusercontent.com/notAI-tech/NudeNet/6ccc81c6c305cccfd46d92b414f8a5c0a816574d/nudenet/320n.onnx"

_NUDENET = ModelSpec(
    role="nudenet",
    filename="nudenet_320n.onnx",
    url=_NUDENET_URL,
    sha256="c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f",
    precision="fp32",
    architecture="YOLOv8n-NudeNet-v3",
    required=False,
    license="AGPL-3.0",
)

_BISENET = ModelSpec(
    role="face_parser",
    filename="face_parsing_bisenet_resnet18.onnx",
    url="https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx",
    sha256="0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
    precision="fp32",
    architecture="BiSeNet-ResNet18",
    license="MIT",
)

# InsightFace ArcFace R50 (buffalo_l `w600k_r50.onnx`). Its license restricts
# use to non-commercial research, so — like MobileCLIP — it ships with no
# download URL: the user extracts it from buffalo_l and places it manually.
_ARCFACE_R50 = ModelSpec(
    role="face_recognizer",
    filename="w600k_r50.onnx",
    url=None,
    sha256=None,
    precision="fp32",
    architecture="ArcFace-R50-w600k",
    license="InsightFace (non-commercial research)",
    recognizer_kind="arcface",
)


MODEL_PROFILES: dict[str, ModelProfile] = {
    "stable": ModelProfile(
        name="stable",
        description="Validated FP32 OpenCV-DNN models.",
        recommended_min_similarity=0.50,
        legacy_fallback=True,
        models={
            "face_detector": ModelSpec(
                "face_detector", "face_detection_yunet_2023mar.onnx",
                _zoo_url("face_detection_yunet/face_detection_yunet_2023mar.onnx"),
                "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                "fp32", "YuNet-2023mar", license="MIT",
            ),
            "face_recognizer": ModelSpec(
                "face_recognizer", "face_recognition_sface_2021dec.onnx",
                _zoo_url("face_recognition_sface/face_recognition_sface_2021dec.onnx"),
                "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
                "fp32", "SFace-MobileFaceNet", license="Apache-2.0",
            ),
            "face_parser": _BISENET,
            "person_detector": ModelSpec(
                "person_detector", "person_detection_mediapipe_2023mar.onnx",
                _zoo_url("person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx"),
                "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f",
                "fp32", "MediaPipe-PersonDet", license="Apache-2.0",
            ),
            "pose": ModelSpec(
                "pose", "pose_estimation_mediapipe_2023mar.onnx",
                _zoo_url("pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx"),
                "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f",
                "fp32", "MediaPipe-Pose", license="Apache-2.0",
            ),            "appearance": ModelSpec(
                "appearance", "mobileclip_s0_image.onnx", None, None,
                "fp32", "MobileCLIP-S0-image", required=False,
                license="Apple ML Research Model TOU",
            ),
            "nudenet": _NUDENET,
        },
    ),
    "experimental": ModelProfile(
        name="experimental",
        description="Stable FP32 stack with the InsightFace ArcFace R50 (buffalo_l w600k) recognizer for identity evaluation.",
        recommended_min_similarity=0.40,
        models={
            # Detector, parser, person, pose and appearance match the stable
            # profile so a stable-vs-experimental benchmark isolates the single
            # change under evaluation: the SFace -> ArcFace R50 recognizer swap.
            "face_detector": ModelSpec(
                "face_detector", "face_detection_yunet_2023mar.onnx",
                _zoo_url("face_detection_yunet/face_detection_yunet_2023mar.onnx"),
                "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                "fp32", "YuNet-2023mar", license="MIT",
            ),
            "face_recognizer": _ARCFACE_R50,
            "face_parser": _BISENET,
            "person_detector": ModelSpec(
                "person_detector", "person_detection_mediapipe_2023mar.onnx",
                _zoo_url("person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx"),
                "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f",
                "fp32", "MediaPipe-PersonDet", license="Apache-2.0",
            ),
            "pose": ModelSpec(
                "pose", "pose_estimation_mediapipe_2023mar.onnx",
                _zoo_url("pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx"),
                "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f",
                "fp32", "MediaPipe-Pose", license="Apache-2.0",
            ),
            "appearance": ModelSpec(
                "appearance", "mobileclip_s0_image.onnx", None, None,
                "fp32", "MobileCLIP-S0-image", required=False,
                license="Apple ML Research Model TOU",
            ),
            "nudenet": _NUDENET,
        },
    ),
}


def get_profile(name: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model profile: {name}") from exc


def model_manifest(profile: ModelProfile, models_dir: Path, checksum: Any) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for role, spec in profile.models.items():
        path = profile.path(models_dir, role)
        exists = path.is_file()
        result[role] = {
            "architecture": spec.architecture,
            "precision": spec.precision,
            "filename": spec.filename,
            "path": str(path.resolve()),
            "sha256": checksum(path) if exists else None,
            "expected_sha256": spec.sha256,
            "required": spec.required,
            "available": exists,
            "license": spec.license,
            "input_name": spec.input_name,
        }
    return result
