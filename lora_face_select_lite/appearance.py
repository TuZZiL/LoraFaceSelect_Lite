from __future__ import annotations

from pathlib import Path
from typing import Any


class MobileCLIPAppearanceBackend:
    """Image-only MobileCLIP-S0 ONNX reranker through OpenCV DNN.

    The model is optional because Apple's checkpoint license is restricted to
    research/non-commercial use. This backend never replaces SFace identity.
    """

    def __init__(self, model_path: str | Path, *, architecture: str = "MobileCLIP-S0-image") -> None:
        import cv2

        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"Optional MobileCLIP image model not found: {path}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.architecture = architecture
        self.provider = f"{architecture}/OpenCV-DNN-CPU"

    @staticmethod
    def _square_crop(image: Any, bbox: tuple[float, float, float, float], factor: float, y_shift: float = 0.0) -> Any:
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        side = max(x2 - x1, y2 - y1) * factor
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2 + y_shift * (y2 - y1)
        left, top = max(0, round(cx - side / 2)), max(0, round(cy - side / 2))
        right, bottom = min(width, round(cx + side / 2)), min(height, round(cy + side / 2))
        return image[top:bottom, left:right]

    def _embed(self, image_bgr: Any) -> Any:
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        self.net.setInput(blob)
        embedding = self.net.forward().reshape(-1).astype(np.float32)
        return embedding / max(1e-12, float(np.linalg.norm(embedding)))

    def embed(self, image_bgr: Any, bbox: tuple[float, float, float, float]) -> tuple[Any, Any]:
        face = self._square_crop(image_bgr, bbox, 1.20)
        head = self._square_crop(image_bgr, bbox, 1.90, -0.18)
        if face.size == 0 or head.size == 0:
            raise ValueError("Cannot create MobileCLIP face/head crop")
        return self._embed(face), self._embed(head)
