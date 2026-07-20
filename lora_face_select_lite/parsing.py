from __future__ import annotations

from pathlib import Path
from typing import Any


PARSING_LABELS = (
    "background", "skin", "left_brow", "right_brow", "left_eye",
    "right_eye", "glasses", "left_ear", "right_ear", "earring", "nose",
    "mouth", "upper_lip", "lower_lip", "neck", "necklace", "clothes",
    "hair", "hat",
)


class BiSeNetFaceParser:
    """BiSeNet ResNet18 face parser running through OpenCV DNN on CPU."""

    def __init__(self, model_path: str | Path) -> None:
        import cv2

        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"BiSeNet model not found: {path}. Run download-models.")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.provider = "BiSeNet-ResNet18/OpenCV-DNN-CPU"

    def predict(self, image_bgr: Any) -> Any:
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        resized = (resized - np.asarray((0.485, 0.456, 0.406), np.float32)) / np.asarray((0.229, 0.224, 0.225), np.float32)
        blob = np.transpose(resized, (2, 0, 1))[None].astype(np.float32)
        self.net.setInput(blob)
        output = self.net.forward("output")
        mask = output[0].argmax(axis=0).astype(np.uint8)
        return cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


def label_ratios(mask: Any) -> dict[str, float]:
    import numpy as np

    total = max(1, int(mask.size))
    values, counts = np.unique(mask, return_counts=True)
    return {PARSING_LABELS[int(value)]: int(count) / total for value, count in zip(values, counts) if int(value) < len(PARSING_LABELS)}
