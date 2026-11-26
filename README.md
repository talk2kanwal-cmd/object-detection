# Computer Vision Object Detection

A demonstrable computer-vision MVP for detecting multiple objects in images and videos. It provides a FastAPI REST API, a Streamlit interface, and a Docker Compose setup. The detector uses the pretrained Ultralytics YOLOv8 nano model by default.

## Tools and technologies

| Tool | Purpose |
| --- | --- |
| Python | Application and API implementation |
| Ultralytics YOLOv8 | Object detection model |
| OpenCV, NumPy, Pillow | Video/image decoding and image processing |
| FastAPI, Uvicorn, python-multipart | REST API and multipart file uploads |
| Streamlit, pandas, PyArrow | Interactive interface and detection result tables |
| Requests | Streamlit-to-API communication |
| pytest | Automated tests |
| Docker, Docker Compose | Container packaging and local multi-service deployment |

## What works in this phase

- Upload an image and view detected labels, confidence scores, and bounding boxes.
- Upload a video and process every readable frame, returning detections with frame indexes and timestamps.
- Call the same image and video detection features through documented REST endpoints.
- Run the API and UI as separate services with Docker Compose.

## Run locally

Use Python 3.10 to 3.12 (Python 3.12 is recommended). Install the packages and start the API and UI in separate terminals:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn api:app --reload
```

In another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open the UI at <http://localhost:8501>. The first detection downloads `yolov8n.pt`, so the API needs network access initially. Set `YOLO_MODEL` to a local Ultralytics-compatible model path to use a different model. Set `DETECTION_API_URL` if the API is not at `http://localhost:8000`.

## Run with Docker

```powershell
docker compose up --build
```

Open <http://localhost:8501> for the UI, <http://localhost:8000/docs> for interactive API documentation, or <http://localhost:8000/health> for the health check. The model is downloaded when the first detection request is made.

## REST API

Both detection endpoints accept multipart form data with a `file` field and an optional `confidence` query parameter (0 to 1, default 0.25).

### Image

```powershell
curl.exe -X POST "http://localhost:8000/detect/image?confidence=0.3" -F "file=@image.jpg"
```

Returns the original filename, image dimensions, and a `detections` array. Each detection has a `label`, `confidence`, `class_id`, and `box` with pixel coordinates `x1`, `y1`, `x2`, and `y2`.

### Video

```powershell
curl.exe -X POST "http://localhost:8000/detect/video?confidence=0.3" -F "file=@video.mp4"
```

Returns video dimensions, FPS, frame count, and a `frames` array. Each frame has a zero-based `frame_index`, `timestamp_seconds`, and its `detections`. Supported video extensions are MP4, MOV, AVI, and MKV. Processing every frame can be computationally expensive and responses grow with video length; this MVP does not impose an upload-size or duration limit.

## Tests

```powershell
python -m pip install pytest
python -m pytest
```

The unit tests use a stub detector model and do not download model weights.

## Cloud deployment

Build the Docker image and deploy it to a container platform that supports Docker Compose or separate container services. Expose port 8501 for the UI and port 8000 for the API; configure the Streamlit service's `DETECTION_API_URL` to the reachable API URL. Configure `YOLO_MODEL` and persistent model storage as appropriate for the selected platform. For a public deployment, add authentication, request-size and video-duration limits, TLS, and resource monitoring before accepting untrusted uploads.

## Licensing note

Ultralytics software and pretrained weights have licensing terms that may affect commercial use. Confirm the applicable license for the intended client deployment before release.
