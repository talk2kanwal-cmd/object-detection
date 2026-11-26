from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from detector import ObjectDetector

client = TestClient(api.app)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[1, 2, 30, 40]])
    conf = FakeTensor([0.91])
    cls = FakeTensor([0])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    names = {0: "person"}

    def predict(self, **kwargs):
        self.confidence = kwargs["conf"]
        return [FakeResult()]


def build_detector():
    detector = ObjectDetector.__new__(ObjectDetector)
    detector.model = FakeModel()
    return detector


def test_detect_frame_returns_label_score_and_box():
    detector = build_detector()

    detections = detector.detect_frame(np.zeros((48, 48, 3)), confidence=0.4)

    assert detections == [
        {
            "label": "person",
            "confidence": 0.91,
            "class_id": 0,
            "box": {"x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0},
        }
    ]
    assert detector.model.confidence == 0.4


def test_detect_frame_rejects_invalid_confidence():
    detector = build_detector()

    with pytest.raises(ValueError, match="between 0 and 1"):
        detector.detect_frame(np.zeros((48, 48, 3)), confidence=1.1)


def test_image_endpoint_returns_detections(monkeypatch):
    class StubDetector:
        def detect_frame(self, image_rgb, confidence):
            assert image_rgb.shape == (4, 6, 3)
            assert confidence == 0.35
            return build_detector().detect_frame(image_rgb, confidence)

    image_bytes = BytesIO()
    Image.new("RGB", (6, 4)).save(image_bytes, format="PNG")
    monkeypatch.setattr(api, "get_detector", lambda: StubDetector())

    response = client.post(
        "/detect/image",
        params={"confidence": 0.35},
        files={"file": ("sample.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["width"] == 6
    assert response.json()["height"] == 4
    assert response.json()["detections"][0]["label"] == "person"


def test_image_endpoint_rejects_invalid_image():
    response = client.post(
        "/detect/image",
        files={"file": ("sample.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
