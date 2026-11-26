from functools import lru_cache
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
import numpy as np
from PIL import Image, UnidentifiedImageError

from detector import ObjectDetector

app = FastAPI(
    title="Computer Vision Detection API",
    description="Detect objects in uploaded images and videos.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_detector() -> ObjectDetector:
    return ObjectDetector()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect/image")
def detect_image(
    file: Annotated[UploadFile, File(...)],
    confidence: Annotated[float, Query(ge=0, le=1)] = 0.25,
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    try:
        with Image.open(file.file) as image:
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
            image_array = np.asarray(image_rgb)
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded image is invalid.") from error

    detections = get_detector().detect_frame(image_array, confidence)
    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "detections": detections,
    }


@app.post("/detect/video")
def detect_video(
    file: Annotated[UploadFile, File(...)],
    confidence: Annotated[float, Query(ge=0, le=1)] = 0.25,
) -> dict:
    allowed_extensions = {".avi", ".mkv", ".mov", ".mp4"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Upload a video with one of these extensions: {', '.join(sorted(allowed_extensions))}.",
        )

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = temporary_file.name
            shutil.copyfileobj(file.file, temporary_file)
        results = get_detector().detect_video(temporary_path, confidence)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)

    return {"filename": file.filename, **results}
