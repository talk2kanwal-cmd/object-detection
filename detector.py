import os
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np


class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(TypedDict):
    label: str
    confidence: float
    class_id: int
    box: BoundingBox


class ObjectDetector:
    def __init__(self, model_path: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path or os.getenv("YOLO_MODEL", "yolov8n.pt"))

    def detect_frame(
        self, image_rgb: np.ndarray, confidence: float = 0.25
    ) -> list[Detection]:
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        result = self.model.predict(
            source=image_rgb, conf=confidence, verbose=False
        )[0]
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        detections: list[Detection] = []

        for coordinates, score, raw_class_id in zip(boxes, scores, class_ids):
            class_id = int(raw_class_id)
            names = self.model.names
            label = names[class_id] if isinstance(names, list) else names[class_id]
            x1, y1, x2, y2 = (float(value) for value in coordinates)
            detections.append(
                {
                    "label": str(label),
                    "confidence": float(score),
                    "class_id": class_id,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
        return detections

    def detect_video(
        self, video_path: str | Path, confidence: float = 0.25
    ) -> dict:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("The uploaded file could not be opened as a video.")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        frame_index = 0

        try:
            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break
                detections = self.detect_frame(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), confidence
                )
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": (
                            frame_index / fps if fps > 0 else None
                        ),
                        "detections": detections,
                    }
                )
                frame_index += 1
        finally:
            capture.release()

        if frame_index == 0:
            raise ValueError("The uploaded video contains no readable frames.")

        return {
            "width": width,
            "height": height,
            "fps": fps or None,
            "frame_count": frame_index,
            "frames": frames,
        }
