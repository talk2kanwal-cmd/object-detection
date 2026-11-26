import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.getenv("DETECTION_API_URL", "http://localhost:8000").rstrip("/")
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "mkv"]

st.set_page_config(page_title="Object Detection", layout="wide")
st.title("Computer Vision Object Detection")
st.write("Upload an image or video to identify objects, bounding boxes, and confidence scores.")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Minimum confidence", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )
    st.caption(f"Detection API: {API_URL}")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        health_response.raise_for_status()
        st.success("Detection API is connected.")
    except requests.RequestException as error:
        st.error(f"Detection API is unavailable: {error}")

uploaded_file = st.file_uploader(
    "Choose an image or video",
    type=IMAGE_EXTENSIONS + VIDEO_EXTENSIONS,
)

if uploaded_file is not None and st.button("Detect objects", type="primary"):
    file_bytes = uploaded_file.getvalue()
    is_video = uploaded_file.name.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS
    endpoint = "video" if is_video else "image"
    try:
        response = requests.post(
            f"{API_URL}/detect/{endpoint}",
            params={"confidence": confidence},
            files={
                "file": (
                    uploaded_file.name,
                    file_bytes,
                    uploaded_file.type or "application/octet-stream",
                )
            },
            timeout=600,
        )
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as error:
        detail = error.response.text if error.response is not None else str(error)
        st.error(f"Detection failed: {detail}")
    except ValueError as error:
        st.error(f"The detection API returned an invalid response: {error}")
    else:
        if is_video:
            st.subheader("Video")
            st.video(file_bytes)
            st.caption(
                f"{results['frame_count']} frames processed at "
                f"{results['fps'] or 'unknown'} FPS "
                f"({results['width']} x {results['height']})."
            )
            records = [
                {
                    "Frame": frame["frame_index"],
                    "Timestamp (s)": frame["timestamp_seconds"],
                    "Label": detection["label"],
                    "Confidence": detection["confidence"],
                    "Bounding box (x1, y1, x2, y2)": tuple(
                        round(detection["box"][key], 1)
                        for key in ("x1", "y1", "x2", "y2")
                    ),
                }
                for frame in results["frames"]
                for detection in frame["detections"]
            ]
            st.subheader("Frame-by-frame detections")
            if records:
                st.dataframe(pd.DataFrame(records), width="stretch")
            else:
                st.info("No objects were detected in this video.")
        else:
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
            draw = ImageDraw.Draw(image)
            detections = results["detections"]
            for detection in detections:
                box = detection["box"]
                coordinates = (
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                )
                draw.rectangle(coordinates, outline="red", width=3)
                draw.text(
                    (box["x1"], max(0, box["y1"] - 18)),
                    f"{detection['label']} {detection['confidence']:.2f}",
                    fill="red",
                )
            st.subheader("Image detections")
            st.image(image, caption=uploaded_file.name, width="stretch")
            if detections:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Label": detection["label"],
                                "Confidence": detection["confidence"],
                                "Bounding box (x1, y1, x2, y2)": tuple(
                                    round(detection["box"][key], 1)
                                    for key in ("x1", "y1", "x2", "y2")
                                ),
                            }
                            for detection in detections
                        ]
                    ),
                    width="stretch",
                )
            else:
                st.info("No objects were detected in this image.")
