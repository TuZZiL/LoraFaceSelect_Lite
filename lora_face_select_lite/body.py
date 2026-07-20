"""MediaPipe person detection and pose estimation through OpenCV DNN.

Pre/post-processing follows the Apache-2.0 OpenCV Zoo implementations:
https://github.com/opencv/opencv_zoo/tree/main/models/person_detection_mediapipe
https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BodyObservation:
    bbox: tuple[float, float, float, float]
    landmarks: Any  # 33 rows: x, y, z, visibility, presence
    mask: Any
    confidence: float
    detection_score: float


def _anchors() -> Any:
    import numpy as np

    values = []
    for grid, repeats in ((28, 2), (14, 2), (7, 6)):
        for y in range(grid):
            for x in range(grid):
                values.extend([((x + 0.5) / grid, (y + 0.5) / grid)] * repeats)
    return np.asarray(values, dtype=np.float32)


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-6, left_area + right_area - intersection)


class MediaPipeBodyBackend:
    def __init__(self, detector_path: str | Path, pose_path: str | Path, *, detector_threshold: float = 0.35, pose_threshold: float = 0.10) -> None:
        import cv2

        detector_path, pose_path = Path(detector_path), Path(pose_path)
        for path in (detector_path, pose_path):
            if not path.is_file():
                raise RuntimeError(f"Body model not found: {path}. Run download-models.")
        self.detector = cv2.dnn.readNet(str(detector_path))
        self.pose = cv2.dnn.readNet(str(pose_path))
        self.detector_threshold = detector_threshold
        self.pose_threshold = pose_threshold
        self.anchors = _anchors()
        self.provider = "MediaPipe-PersonDet+Pose/OpenCV-DNN-CPU"

    def _detect(self, image: Any) -> list[Any]:
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        ratio = min(224 / height, 224 / width)
        resized_w, resized_h = max(1, int(width * ratio)), max(1, int(height * ratio))
        resized = cv2.resize(rgb, (resized_w, resized_h))
        left, top = (224 - resized_w) // 2, (224 - resized_h) // 2
        padded = cv2.copyMakeBorder(resized, top, 224 - resized_h - top, left, 224 - resized_w - left, cv2.BORDER_CONSTANT, value=0)
        self.detector.setInput(np.transpose(padded, (2, 0, 1))[None])
        outputs = self.detector.forward(self.detector.getUnconnectedOutLayersNames())
        regression = next(output for output in outputs if output.shape[-1] == 12)[0]
        logits = next(output for output in outputs if output.shape[-1] == 1)[0, :, 0]
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -100, 100)))
        scale = max(width, height)
        centers = regression[:, :2] / 224 + self.anchors
        sizes = regression[:, 2:4] / 224
        boxes = np.c_[centers - sizes / 2, centers + sizes / 2] * scale
        pad = np.asarray((left / ratio, top / ratio, left / ratio, top / ratio))
        boxes -= pad
        landmarks = regression[:, 4:].reshape(-1, 4, 2) / 224 + self.anchors[:, None, :]
        landmarks = landmarks * scale - np.asarray((left / ratio, top / ratio))
        candidates = []
        for index in np.flatnonzero(scores >= self.detector_threshold):
            candidates.append(np.r_[boxes[index], landmarks[index].reshape(-1), scores[index]])
        candidates.sort(key=lambda row: -float(row[-1]))
        kept = []
        for candidate in candidates:
            box = tuple(map(float, candidate[:4]))
            if all(_iou(box, tuple(map(float, other[:4]))) < 0.30 for other in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _match_person(persons: list[Any], face_bbox: tuple[float, float, float, float]) -> Any | None:
        import math

        if not persons:
            return None
        fx, fy = (face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2
        face_size = max(1.0, face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1])

        def score(person: Any) -> float:
            detector_face = tuple(map(float, person[:4]))
            px, py = (detector_face[0] + detector_face[2]) / 2, (detector_face[1] + detector_face[3]) / 2
            distance = math.hypot(fx - px, fy - py) / face_size
            return 4.0 * _iou(face_bbox, detector_face) + math.exp(-0.5 * distance) + 0.1 * float(person[-1])

        winner = max(persons, key=score)
        detector_face = tuple(map(float, winner[:4]))
        # Do not attach a distant body to the target face in group photos.
        expanded = (detector_face[0] - face_size, detector_face[1] - face_size, detector_face[2] + face_size, detector_face[3] + face_size)
        if not (expanded[0] <= fx <= expanded[2] and expanded[1] <= fy <= expanded[3]):
            return None
        return winner

    def analyze(self, image: Any, face_bbox: tuple[float, float, float, float]) -> BodyObservation | None:
        person = self._match_person(self._detect(image), face_bbox)
        if person is None:
            return None
        result = self._estimate_pose(image, person)
        if result is None:
            return None
        bbox, landmarks, mask, confidence = result
        return BodyObservation(tuple(map(float, bbox)), landmarks[:33], mask, confidence, float(person[-1]))

    def _estimate_pose(self, image: Any, person: Any) -> tuple[Any, Any, Any, float] | None:
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        keypoints = person[4:12].reshape(-1, 2).astype(np.float64)
        hip, full_body = keypoints[0], keypoints[1]
        radius = float(np.linalg.norm(hip - full_body))
        if radius < 2:
            return None
        full_box = np.asarray((hip - radius, hip + radius), dtype=np.int32)
        clipped = full_box.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
        if clipped[1, 0] <= clipped[0, 0] or clipped[1, 1] <= clipped[0, 1]:
            return None
        crop = image[clipped[0, 1]:clipped[1, 1], clipped[0, 0]:clipped[1, 0]]
        left, top = clipped[0] - full_box[0]
        right, bottom = full_box[1] - clipped[1]
        crop = cv2.copyMakeBorder(crop, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=0)
        pad_bias = clipped[0] - (left, top)
        local_hip, local_full = hip - pad_bias, full_body - pad_bias
        radians = np.pi / 2 - np.arctan2(-(local_full[1] - local_hip[1]), local_full[0] - local_hip[0])
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = float(np.rad2deg(radians))
        rotation = cv2.getRotationMatrix2D(tuple(local_hip), angle, 1.0)
        rotated = cv2.warpAffine(crop, rotation, (crop.shape[1], crop.shape[0]))
        blob = cv2.cvtColor(cv2.resize(rotated, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.pose.setInput(blob[None])
        outputs = self.pose.forward(self.pose.getUnconnectedOutLayersNames())
        by_shape = {output.shape[-1]: output for output in outputs if output.ndim == 2}
        landmarks = by_shape[195][0].reshape(-1, 5)
        confidence = float(by_shape[1][0, 0])
        if confidence < self.pose_threshold:
            return None
        mask_raw = next(output for output in outputs if output.shape == (1, 256, 256, 1))[0, :, :, 0]
        landmarks[:, 3:] = 1.0 / (1.0 + np.exp(-landmarks[:, 3:]))
        scale = np.asarray((crop.shape[1] / 256, crop.shape[0] / 256))
        centered = (landmarks[:, :2] - 128) * scale
        coordinate_rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
        rotated_landmarks = centered @ coordinate_rotation
        rotation_component = np.asarray(((rotation[0, 0], rotation[1, 0]), (rotation[0, 1], rotation[1, 1])))
        translation = rotation[:, 2]
        inverse_translation = -rotation_component @ translation
        inverse = np.c_[rotation_component, inverse_translation]
        center = np.asarray((crop.shape[1] / 2, crop.shape[0] / 2, 1))
        original_center = inverse @ center
        landmarks[:, :2] = rotated_landmarks + original_center + pad_bias
        visible = landmarks[:33, 4] >= 0.35
        coords = landmarks[:33, :2][visible]
        if not len(coords):
            return None
        bbox = (
            float(np.clip(coords[:, 0].min(), 0, width)),
            float(np.clip(coords[:, 1].min(), 0, height)),
            float(np.clip(coords[:, 0].max(), 0, width)),
            float(np.clip(coords[:, 1].max(), 0, height)),
        )
        inverse_mask_rotation = cv2.getRotationMatrix2D((128, 128), -angle, 1.0)
        body_mask = cv2.warpAffine(mask_raw, inverse_mask_rotation, (256, 256))
        body_mask = cv2.resize(body_mask, (crop.shape[1], crop.shape[0]))
        canvas = np.zeros((height, width), dtype=np.uint8)
        src_x1, src_y1 = max(0, -int(pad_bias[0])), max(0, -int(pad_bias[1]))
        dst_x1, dst_y1 = max(0, int(pad_bias[0])), max(0, int(pad_bias[1]))
        copy_w = min(width - dst_x1, body_mask.shape[1] - src_x1)
        copy_h = min(height - dst_y1, body_mask.shape[0] - src_y1)
        if copy_w > 0 and copy_h > 0:
            canvas[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = (body_mask[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w] > 0).astype(np.uint8) * 255
        return bbox, landmarks, canvas, confidence
