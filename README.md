# VISORA — Models

## Description

VISORA is a security monitoring system for retail stores ("bodegas"). It lets store owners and administrators watch live camera feeds, review detection events and alerts, and manage stores, cameras and users from a single dashboard.

This repository holds the **AI worker pipelines**: standalone Python scripts that process video from a camera in real time and are designed to report detections to the [VISORA backend](https://github.com/Visora-live/visora-backend) via REST API, for display in the [VISORA frontend](https://github.com/Visora-live/visora-frontend).

Intended integration:

```
RTSP camera
    │
    ▼
Worker (this repo)
    │  POST /api/events
    │  POST /api/alerts
    │  POST /api/identifications
    ▼
visora-backend
    │
    ▼
visora-frontend
```

**Current status:** the two packages in this repository (`package_weapon_webcam`, `package_face_webcam`) are **standalone local demos**. Each opens a webcam feed with OpenCV, runs its detection pipeline, and prints the result on screen — neither package includes the backend API calls, database, or event flow shown in the diagram above yet. That integration (wiring detections into `POST /api/events` / `/api/alerts` / `/api/identifications`) is not implemented in this repository at this time.

## Packages

### `package_weapon_webcam` — Weapon detection

Real-time firearm detection pipeline:

1. **YOLOv8** — candidate weapon detector.
2. **YOLOv8-Pose** — extracts body keypoints (only run on YOLO candidates), used to check proximity to hands/arms.
3. **NASNetMobile** — validates the candidate crop to reject false positives.

Displays `Sin arma detectada` (no weapon detected), `Candidato rechazado` (candidate rejected), or `ARMA CONFIRMADA | arma=0.xx` (weapon confirmed, with confidence) on screen.

### `package_face_webcam` — Facial identification

Real-time face identification pipeline:

1. **InsightFace** — face detection and embedding generation.
2. **Custom gallery** — compares embeddings against pre-computed centroids for enrolled identities.

Displays `Identificado: ... | score=...` (identified), `Desconocido | score=...` (unknown), or `Rostro no detectado` (no face detected) on screen.

## Dataset Information

Model weights are **not** included in the repository and must be placed manually before running:

| File | Expected path |
|------|----------------|
| `best.pt` (weapon detector) | `package_weapon_webcam/workspace_modelos/content-detector/weights/best.pt` |
| `yolov8n-pose.pt` | `package_weapon_webcam/yolov8n-pose.pt` |
| `nasnetmobile_weapon_validator_fase4_finetune_50k_final.keras` | `package_weapon_webcam/workspace_modelos/models/` |

InsightFace's own models download automatically on first use.

The face-identification demo ships with a small enrolled-identity gallery under `package_face_webcam/`:

- `data/custom_identities.csv` — enrolled identity metadata.
- `reports/custom_face_enrollment/custom_gallery_centroids.csv` — precomputed embedding centroids used for matching (required at runtime).
- `custom_gallery/` — source images backing the gallery.

## Code Information

- **Language:** Python 3.11.
- **Hardware:** an NVIDIA GPU (RTX 3050 or better) is recommended for real-time inference; both demos are currently configured to force CPU execution (`CUDA_VISIBLE_DEVICES=""` / `CTX_ID=-1`) for local testing.
- **Structure:**

  ```
  package_weapon_webcam/
  ├── run_weapon_webcam.py        # demo entry point
  ├── test_yolo_pose_nasnet_webcam.py
  ├── workspace_modelos/          # model weights (see Dataset Information)
  ├── yolov8n-pose.pt
  ├── requirements.txt
  ├── run_windows.ps1 / run_linux.sh
  └── .env / .env.aws             # worker configuration (see Requirements)

  package_face_webcam/
  ├── run_face_webcam.py          # demo entry point
  ├── recognize_custom_face_camera.py
  ├── custom_gallery/ · data/ · reports/   # gallery data (see Dataset Information)
  ├── requirements.txt
  └── run_windows.ps1 / run_linux.sh
  ```

- **Key tunables** (`run_weapon_webcam.py`): `CAMERA_INDEX`, `YOLO_CONF=0.25`, `YOLO_IMGSZ=960`, `POSE_CONF=0.35`, `POSE_IMGSZ=640`, `NASNET_THRESHOLD=0.5`, `HAND_DISTANCE_THRESHOLD=120.0`, `IOU_DUPLICATE_THRESHOLD=0.5`.
- **Key tunables** (`run_face_webcam.py`): `CAMERA_INDEX`, `CTX_ID=-1`, `DET_SIZE=320`, `DET_THRESH=0.2`, `PAD_RATIO=0.35`, `THRESHOLD=0.35`.
- Exit either demo by pressing `Q` in the video window.

## Usage Instructions

Each package is self-contained; set up and run them independently.

**Weapon detection demo:**

```powershell
cd package_weapon_webcam
py -3.11 -m venv .venv_weapon
.\.venv_weapon\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_weapon_webcam.py
```

```bash
cd package_weapon_webcam
python3 -m venv .venv_weapon
source .venv_weapon/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_weapon_webcam.py
```

**Face identification demo:**

```powershell
cd package_face_webcam
py -3.11 -m venv .venv_face
.\.venv_face\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_face_webcam.py
```

If the camera fails to open, change `CAMERA_INDEX = 0` to `1` or `2` inside the corresponding script.

Both packages also ship `run_windows.ps1` / `run_linux.sh` convenience launchers.

## Requirements

- Python 3.11
- Model weights placed in the paths listed under **Dataset Information**
- Python packages: see each package's `requirements.txt` (OpenCV, Ultralytics YOLOv8, TensorFlow/Keras, InsightFace, and their dependencies)
- For the intended (not-yet-wired) integration with the backend, each worker expects the following environment variables:

  ```env
  CAMERA_ID=1
  RTSP_URL=rtsp://localhost:8554/cam1
  API_BASE=http://localhost:8000/api
  VISORA_USER=admin
  VISORA_PASS=your_password
  FRAME_SKIP=1
  ALERT_COOLDOWN=30
  CAPTURE_WINDOW=1.0
  ```

## Methodology

**Weapon detection pipeline** (`package_weapon_webcam`):

1. Capture a frame from the webcam/RTSP source with OpenCV.
2. Run YOLOv8 over the frame to find weapon candidates.
3. Run YOLOv8-Pose to locate body keypoints (hands/arms).
4. Filter candidates by proximity/intersection with hand/arm zones.
5. Validate the surviving candidate crop with NASNetMobile.
6. Render the result on screen (rejected / confirmed, with confidence).

**Face identification pipeline** (`package_face_webcam`):

1. Capture a frame from the webcam with OpenCV.
2. Detect faces with InsightFace.
3. Generate an embedding for each detected face.
4. Compare the embedding against the pre-loaded gallery centroids.
5. Render the closest match (or "unknown") on screen.

## Citations

This project uses the following third-party models and libraries:

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* [Software]. https://github.com/ultralytics/ultralytics
- Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). *RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild*. CVPR. (via the [InsightFace](https://github.com/deepinsight/insightface) project)
- Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). *Learning Transferable Architectures for Scalable Image Recognition* (NASNet). CVPR.

## License & Contribution Guidelines

Licensed under the [MIT License](./LICENSE).

This is an academic project (TP1). Contributions are currently managed directly by the project author; open an issue or pull request on GitHub if you'd like to propose a change.
