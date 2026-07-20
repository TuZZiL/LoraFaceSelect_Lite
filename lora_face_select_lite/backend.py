from __future__ import annotations

import math
from typing import Any

from .models import FaceObservation

# Canonical InsightFace 5-point template for a 112x112 ArcFace crop
# (right eye, left eye, nose, right mouth corner, left mouth corner). YuNet
# reports its five landmarks in the same order, so the correspondence is
# positional and no reordering is required.
_ARCFACE_TEMPLATE = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def _pose_from_landmarks(landmarks: Any, _image_size: tuple[int, int]) -> tuple[float | None, float | None, float | None]:
    """Estimate a coarse pose from YuNet's five 2D landmarks.

    Five points are insufficient for a reliable 3D solvePnP estimate. Relative
    nose displacement is intentionally used only for the broad MVP bins.
    """
    try:
        import numpy as np

        points = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
        if len(points) < 5:
            return None, None, None
        first_eye, second_eye, nose, first_mouth, second_mouth = points[:5]
        eye_vector = second_eye - first_eye
        eye_distance = float(np.linalg.norm(eye_vector))
        eye_midpoint = (first_eye + second_eye) * 0.5
        mouth_midpoint = (first_mouth + second_mouth) * 0.5
        eye_to_mouth = float(mouth_midpoint[1] - eye_midpoint[1])
        if eye_distance < 1.0 or abs(eye_to_mouth) < 1.0:
            return None, None, None
        facial_midpoint = (eye_midpoint + mouth_midpoint) * 0.5
        yaw = float(np.clip((nose[0] - facial_midpoint[0]) / eye_distance * 90.0, -75.0, 75.0))
        nose_fraction = (nose[1] - eye_midpoint[1]) / eye_to_mouth
        pitch = float(np.clip((nose_fraction - 0.5) * 90.0, -45.0, 45.0))
        roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))
        return yaw, pitch, roll
    except Exception:
        return None, None, None


class OpenCVYuNetSFaceBackend:
    """Face detector/recognizer using only OpenCV DNN on CPU by default.

    The recognizer supports two families selected by ``recognizer_kind``:

    * ``sface`` (default) uses OpenCV's ``FaceRecognizerSF`` for alignment and
      the 128-d MobileFaceNet embedding, optionally through a named DNN input
      for the INT8BQ export.
    * ``arcface`` aligns the face to the canonical InsightFace 5-point template
      and runs an ArcFace ONNX (e.g. buffalo_l ``w600k_r50``) through OpenCV DNN
      to produce a 512-d embedding.
    """

    def __init__(
        self,
        detector_model: str,
        recognizer_model: str,
        *,
        detector_score: float = 0.85,
        nms_threshold: float = 0.3,
        profile: str = "stable",
        precision: str = "fp32",
        recognizer_input_name: str | None = None,
        recognizer_kind: str = "sface",
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is missing. Install the project dependencies first.") from exc
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Installed OpenCV does not expose FaceDetectorYN; install opencv-python-headless>=4.10.")
        self.recognizer_kind = recognizer_kind
        if recognizer_kind == "sface" and not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("Installed OpenCV does not expose FaceRecognizerSF; install opencv-python-headless>=4.10.")
        from pathlib import Path
        detector_path, recognizer_path = Path(detector_model), Path(recognizer_model)
        if not detector_path.is_file():
            raise RuntimeError(f"YuNet model not found: {detector_path}. Run download-models.")
        if not recognizer_path.is_file():
            raise RuntimeError(
                f"Face recognizer model not found: {recognizer_path}. Run download-models, "
                "or place the manual model (see README) for this profile."
            )
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), detector_score, nms_threshold, 5000)
        self.recognizer = None
        self.feature_net = None
        self.recognizer_net = None
        self.recognizer_input_name = recognizer_input_name
        if recognizer_kind == "arcface":
            import numpy as np

            self.recognizer_net = cv2.dnn.readNetFromONNX(str(recognizer_path))
            self._arcface_template = np.asarray(_ARCFACE_TEMPLATE, dtype=np.float32)
            self.recognizer_net.setInput(np.zeros((1, 3, 112, 112), dtype=np.float32))
            if self.recognizer_net.forward().size == 0:
                raise RuntimeError(f"ArcFace model produced an empty feature vector: {recognizer_path}")
            recognizer_label = "ArcFace"
        else:
            self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
            self.feature_net = cv2.dnn.readNetFromONNX(str(recognizer_path)) if recognizer_input_name else None
            if self.feature_net is not None:
                import numpy as np

                self.feature_net.setInput(np.zeros((1, 3, 112, 112), dtype=np.float32), recognizer_input_name)
                if self.feature_net.forward().size == 0:
                    raise RuntimeError(f"SFace model produced an empty feature vector: {recognizer_path}")
            recognizer_label = "SFace"
        self.provider = f"YuNet+{recognizer_label}/OpenCV-DNN-CPU ({profile}, {precision})"

    def _embed(self, image_bgr: Any, row: Any, landmarks: Any) -> Any:
        import cv2

        if self.recognizer_kind == "arcface":
            return self._embed_arcface(image_bgr, landmarks)
        aligned = self.recognizer.alignCrop(image_bgr, row)
        if self.feature_net is None:
            return self.recognizer.feature(aligned)
        # The official INT8BQ SFace export has unused legacy graph inputs. A
        # named input avoids an ambiguous empty-name call.
        blob = cv2.dnn.blobFromImage(aligned, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        self.feature_net.setInput(blob, self.recognizer_input_name)
        return self.feature_net.forward()

    def _embed_arcface(self, image_bgr: Any, landmarks: Any) -> Any:
        import cv2
        import numpy as np

        source = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        matrix, _ = cv2.estimateAffinePartial2D(source, self._arcface_template, method=cv2.LMEDS)
        if matrix is None:
            raise RuntimeError("Cannot align face to the ArcFace template")
        aligned = cv2.warpAffine(image_bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0.0)
        # InsightFace ArcFace expects RGB normalized as (pixel - 127.5) / 127.5.
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.recognizer_net.setInput(blob)
        return self.recognizer_net.forward()

    def analyze(self, image_bgr: Any) -> list[FaceObservation]:
        import numpy as np

        height, width = image_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image_bgr)
        if faces is None:
            return []
        observations = []
        for row in np.asarray(faces, dtype=np.float32):
            x, y, w, h = map(float, row[:4])
            landmarks = row[4:14].reshape(5, 2)
            embedding = self._embed(image_bgr, row, landmarks)
            yaw, pitch, roll = _pose_from_landmarks(landmarks, (height, width))
            observations.append(FaceObservation(embedding=embedding, bbox=(x, y, x + w, y + h), detection_score=float(row[14]), landmarks=landmarks, yaw=yaw, pitch=pitch, roll=roll))
        return observations
