"""NudeNet v3 body-part detector through ONNX Runtime.

The bundled profile uses the official 320n.onnx YOLOv8 model. NudeNet
detects covered/exposed body parts; its confidence is not an anatomical
measurement and must not be used to infer body or chest size.

Model source: https://github.com/notAI-tech/NudeNet/tree/v3
License: AGPL-3.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LABELS = [
    "female-genitalia-covered",
    "female-face",
    "buttocks-exposed",
    "female-breast-exposed",
    "female-genitalia-exposed",
    "male-breast-exposed",
    "anus-exposed",
    "feet-exposed",
    "belly-covered",
    "feet-covered",
    "armpits-covered",
    "armpits-exposed",
    "male-face",
    "belly-exposed",
    "male-genitalia-exposed",
    "anus-covered",
    "female-breast-covered",
    "buttocks-covered",
]


@dataclass
class NudeNetDetection:
    label: str
    class_id: int
    score: float
    box: tuple[int, int, int, int]  # x, y, width, height in full-image pixels


@dataclass
class NudeNetResult:
    detections: list[NudeNetDetection] = field(default_factory=list)
    provider: str = ""

    def detections_by_label(self, label: str) -> list[NudeNetDetection]:
        return [d for d in self.detections if d.label == label]

    def max_score(self, label: str) -> float:
        return max((d.score for d in self.detections_by_label(label)), default=0.0)

    def any_label(self, *labels: str) -> bool:
        return any(self.max_score(label) > 0.0 for label in labels)

    @property
    def labels(self) -> list[str]:
        return sorted({d.label for d in self.detections})

    @property
    def max_detection_score(self) -> float:
        return max((d.score for d in self.detections), default=0.0)


class NudeNetBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: int = 320,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is missing. Reinstall the project dependencies.") from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise RuntimeError(f"NudeNet model not found: {model_path}. Run download-models.")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("NudeNet score_threshold must be between 0 and 1")
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.model_path = model_path
        try:
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize NudeNet ONNX model: {model_path}: {exc}") from exc
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Unexpected NudeNet model inputs/outputs: {len(inputs)}/{len(outputs)}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.provider = "NudeNet-v3-320n/ONNXRuntime-CPU"

    @staticmethod
    def _clip_roi(image: Any, roi: tuple[float, float, float, float] | None) -> tuple[Any, int, int]:
        if roi is None:
            return image, 0, 0
        height, width = image.shape[:2]
        x1, y1, x2, y2 = roi
        # Pose boxes contain visible landmark extrema. A small margin retains
        # body parts near their boundary without including distant people.
        margin_x = max(8.0, (x2 - x1) * 0.12)
        margin_y = max(8.0, (y2 - y1) * 0.08)
        left = max(0, int(round(x1 - margin_x)))
        top = max(0, int(round(y1 - margin_y)))
        right = min(width, int(round(x2 + margin_x)))
        bottom = min(height, int(round(y2 + margin_y)))
        if right <= left or bottom <= top:
            return image[0:0, 0:0], left, top
        return image[top:bottom, left:right], left, top

    def detect(self, image_bgr: Any, roi: tuple[float, float, float, float] | None = None) -> NudeNetResult:
        import cv2
        import numpy as np

        if image_bgr is None or getattr(image_bgr, "ndim", 0) != 3 or image_bgr.shape[2] != 3:
            raise ValueError("NudeNet expects a non-empty BGR image with three channels")
        image, offset_x, offset_y = self._clip_roi(image_bgr, roi)
        if image.size == 0:
            return NudeNetResult(provider=self.provider)

        height, width = image.shape[:2]
        scale = self.input_size / max(height, width)
        resized_width = max(1, min(self.input_size, int(round(width * scale))))
        resized_height = max(1, min(self.input_size, int(round(height * scale))))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = self.input_size - resized_width
        pad_y = self.input_size - resized_height
        pad_left, pad_top = pad_x // 2, pad_y // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_y - pad_top,
            pad_left,
            pad_x - pad_left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, swapRB=True, crop=False)
        raw = np.squeeze(self.session.run([self.output_name], {self.input_name: blob})[0])
        expected_columns = 4 + len(LABELS)
        if raw.ndim != 2:
            raise RuntimeError(f"Unexpected NudeNet output shape: {raw.shape}")
        if raw.shape[0] == expected_columns:
            rows = raw.T
        elif raw.shape[1] == expected_columns:
            rows = raw
        else:
            raise RuntimeError(f"Unexpected NudeNet output shape: {raw.shape}; expected 18 classes")

        detections: list[NudeNetDetection] = []
        for row in rows:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.score_threshold:
                continue
            center_x, center_y, box_width, box_height = map(float, row[:4])
            left = int(round((center_x - box_width / 2 - pad_left) / scale))
            top = int(round((center_y - box_height / 2 - pad_top) / scale))
            right = int(round((center_x + box_width / 2 - pad_left) / scale))
            bottom = int(round((center_y + box_height / 2 - pad_top) / scale))
            left, top = max(0, left), max(0, top)
            right, bottom = min(width, right), min(height, bottom)
            if right <= left or bottom <= top:
                continue
            detections.append(
                NudeNetDetection(
                    LABELS[class_id],
                    class_id,
                    score,
                    (left + offset_x, top + offset_y, right - left, bottom - top),
                )
            )

        if not detections:
            return NudeNetResult(provider=self.provider)
        indices = cv2.dnn.NMSBoxes(
            [list(d.box) for d in detections],
            [d.score for d in detections],
            self.score_threshold,
            self.nms_threshold,
        )
        kept = [detections[int(index)] for index in np.asarray(indices).reshape(-1)]
        return NudeNetResult(kept, provider=self.provider)
