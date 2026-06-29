from __future__ import annotations

import csv
import logging
import os
import sys
import time
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
# UDP = menor latencia en LAN. Si hay pérdida de paquetes frecuente, cambiar a tcp.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0"

import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

import tensorflow as tf
from tensorflow.keras.applications.nasnet import preprocess_input
from insightface.app import FaceAnalysis

# ── Worker config — edit these or set as environment variables ────────────────
CAMERA_ID      = int(os.getenv("CAMERA_ID", "1"))
RTSP_URL       = os.getenv("RTSP_URL", f"rtsp://localhost:8554/cam{CAMERA_ID}")
API_BASE       = os.getenv("API_BASE", "http://localhost:8000/api")
API_USER       = os.getenv("VISORA_USER", "admin")
API_PASS       = os.getenv("VISORA_PASS", "")
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "30"))
HEADLESS       = os.getenv("HEADLESS", "0") == "1"
SNAPSHOT_DIR   = Path(os.getenv("SNAPSHOT_DIR", r"C:\visora_snapshots"))
PAUSE_FILE     = SNAPSHOT_DIR / f"cam{CAMERA_ID}.paused"
FRAME_SKIP     = int(os.getenv("FRAME_SKIP", "1"))   # process 1 of every N frames (1=every frame, GPU recommended)

# ── Structured logger ─────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SNAPSHOT_DIR / f"cam{CAMERA_ID}_diag.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger(f"visora.cam{CAMERA_ID}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

LOG = _setup_logger()

# ── Face ID config ───────────────────────────────────────────────────────────
_DEFAULT_CENTROIDS = (
    Path(__file__).resolve().parents[1]
    / "package_face_webcam" / "reports" / "custom_face_enrollment" / "custom_gallery_centroids.csv"
)
CENTROIDS_PATH  = Path(os.getenv("CENTROIDS_PATH", str(_DEFAULT_CENTROIDS)))
FACE_THRESHOLD  = float(os.getenv("FACE_THRESHOLD", "0.38"))
FACE_DET_SIZE   = 320
FACE_DET_THRESH = 0.35
FACE_CROP_RATIO = 0.40   # top 40% of person bbox = face zone

# ── Detection config ──────────────────────────────────────────────────────────
YOLO_CONF              = 0.35
YOLO_IMGSZ             = int(os.getenv("YOLO_IMGSZ", "640"))   # 640 recommended on GPU; set env to 480 on CPU
POSE_CONF              = 0.35
POSE_IMGSZ             = int(os.getenv("POSE_IMGSZ", "640"))
NASNET_THRESHOLD       = 0.50
HAND_DISTANCE_THRESHOLD = 120.0
KP_CONF_MIN            = 0.25
ARM_ZONE_MARGIN        = 40
IOU_DUPLICATE_THRESHOLD = 0.5
CAPTURE_WINDOW  = float(os.getenv("CAPTURE_WINDOW", "3.0"))  # reduced from 5s — 3s enough for best frame
BLUR_MIN_VAR    = 40.0   # Laplacian variance below this = too blurry, skip frame from buffer
FACE_CROP_PAD   = 15     # extra pixels of padding around face zone crop

CLASS_NAMES = ["arma", "no_arma"]

KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10

# ── Image processing — built once, reused per-crop ────────────────────────────
_CLAHE_CROP     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_CLAHE_FACE     = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
_SHARPEN_KERNEL = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)


# ── Model paths ───────────────────────────────────────────────────────────────

def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_model_paths() -> Dict[str, Path]:
    root = project_root()
    return {
        "yolo_weapon": root / "workspace_modelos" / "content-detector" / "weights" / "best.pt",
        "yolo_pose":   root / "yolov8n-pose.pt",
        "nasnet":      root / "workspace_modelos" / "models" / "nasnetmobile_weapon_validator_fase4_finetune_50k_final.keras",
    }


def validate_model_paths(paths: Dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el modelo `{label}` en: {path}")


# ── Face ID helpers ───────────────────────────────────────────────────────────

def _normalize(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def load_centroids(path: Path) -> List[Dict]:
    if not path.exists():
        LOG.warning("Centroides no encontrados: %s — identificación facial deshabilitada", path)
        return []
    rows: List[Dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals = [float(v) for v in str(row["centroid_vector"]).split("|") if v.strip()]
            edad_raw = row.get("edad", "").strip()
            rows.append({
                "identity_id":      str(row.get("identity_id", "")).strip(),
                "nombre":           str(row.get("nombre", "")).strip() or None,
                "apellido_paterno": str(row.get("apellido_paterno", "")).strip() or None,
                "apellido_materno": str(row.get("apellido_materno", "")).strip() or None,
                "dni":              str(row.get("dni", "")).strip(),
                "edad":             int(edad_raw) if edad_raw.isdigit() else None,
                "centroid":         _normalize(np.array(vals, dtype=np.float32)),
            })
    LOG.info("%d identidades cargadas para reconocimiento facial", len(rows))
    return rows


def match_face(embedding: np.ndarray, centroids: List[Dict]) -> Tuple[Optional[Dict], float]:
    if not centroids:
        return None, 0.0
    emb = _normalize(embedding)
    best, best_score = max(((c, float(np.dot(emb, c["centroid"]))) for c in centroids), key=lambda x: x[1])
    return best, best_score


def find_carrier_person_box(
    weapon_box: Sequence[float], pose_result
) -> Optional[List[float]]:
    """Return the bounding box of the person closest to the weapon center."""
    if pose_result.boxes is None or len(pose_result.boxes) == 0:
        return None
    wcx, wcy = box_center(weapon_box)
    best_box, best_dist = None, float("inf")
    for pb in pose_result.boxes:
        pbox = pb.xyxy[0].cpu().numpy().tolist()
        cx, cy = box_center(pbox)
        dist = math.hypot(cx - wcx, cy - wcy)
        if dist < best_dist:
            best_dist = dist
            best_box = pbox
    return best_box


def crop_face_zone(
    frame: np.ndarray, person_box: Sequence[float], ratio: float = FACE_CROP_RATIO
) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, person_box)
    face_h = max(1, int((y2 - y1) * ratio))
    x1c = max(0, x1 - FACE_CROP_PAD)
    y1c = max(0, y1 - FACE_CROP_PAD)
    x2c = min(w, x2 + FACE_CROP_PAD)
    y2c = min(h, y1c + face_h + FACE_CROP_PAD * 2)
    crop = frame[y1c:y2c, x1c:x2c]
    return crop if crop.size > 0 else None


def _get_embedding(face_obj) -> np.ndarray:
    emb = face_obj.normed_embedding if (hasattr(face_obj, "normed_embedding") and face_obj.normed_embedding is not None) else face_obj.embedding
    return np.asarray(emb, dtype=np.float32)


def identify_carrier(
    frame: np.ndarray,
    weapon_box: Sequence[float],
    face_app: FaceAnalysis,
    centroids: List[Dict],
    pose_result=None,
) -> Optional[Dict]:
    """Identify weapon carrier. Pose-guided crop first, full-frame fallback if needed."""
    def _resolve(emb: np.ndarray) -> Optional[Dict]:
        identity, score = match_face(emb, centroids)
        if identity and score >= FACE_THRESHOLD:
            return {**identity, "face_score": score, "known": True}
        return {"known": False, "face_score": score}

    # Pose-guided: crop top face zone of the person nearest the weapon
    if pose_result is not None:
        person_box = find_carrier_person_box(weapon_box, pose_result)
        if person_box is not None:
            crop = crop_face_zone(frame, person_box)
            if crop is not None:
                faces = face_app.get(enhance_face_crop(crop))
                if faces:
                    return _resolve(_get_embedding(faces[0]))

    # Fallback: all faces in full frame, pick closest to weapon center
    all_faces = face_app.get(frame)
    if not all_faces:
        return None
    wcx, wcy = box_center(weapon_box)
    closest = min(
        all_faces,
        key=lambda f: euclidean_distance(box_center(f.bbox.tolist()), (wcx, wcy)),
    )
    return _resolve(_get_embedding(closest))


# ── Backend integration ───────────────────────────────────────────────────────

def api_login() -> str:
    resp = requests.post(
        f"{API_BASE}/auth/login",
        json={"username_or_email": API_USER, "password": API_PASS},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token", "")
    LOG.info("Login OK como '%s' en %s", API_USER, API_BASE)
    return token


def get_tienda_id(token: str) -> Optional[int]:
    try:
        resp = requests.get(
            f"{API_BASE}/cameras/{CAMERA_ID}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("tienda_id")
    except Exception as exc:
        LOG.warning("No se pudo obtener tienda_id de cámara %d: %s", CAMERA_ID, exc)
    return None


def _with_token_refresh(token_ref: List[str], fn) -> None:
    try:
        fn(token_ref[0])
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403):
            print("[VISORA] Token expirado — renovando...")
            try:
                token_ref[0] = api_login()
                fn(token_ref[0])
            except Exception as e:
                LOG.error("Error tras renovar token: %s", e)
        else:
            LOG.error("HTTP %s al llamar API: %s", exc.response.status_code if exc.response else "?", exc)
    except Exception as exc:
        LOG.error("Error de red: %s", exc)


def post_weapon_initial(
    token_ref: List[str], tienda_id: Optional[int]
) -> Tuple[Optional[int], Optional[int]]:
    """POST event + alert immediately on first weapon confirmation. Returns (event_id, alert_id)."""
    result: List[Tuple[Optional[int], Optional[int]]] = [(None, None)]

    def _post(tok: str) -> None:
        headers = {"Authorization": f"Bearer {tok}"}
        ev = requests.post(
            f"{API_BASE}/events",
            json={
                "tipo": "weapon_detection",
                "severidad": "alta",
                "estado": "abierto",
                "camara_id": CAMERA_ID,
                "comentario": "Arma detectada — analizando incidente...",
            },
            headers=headers,
            timeout=10,
        )
        ev.raise_for_status()
        event_id = ev.json().get("id")

        alert_payload: Dict = {
            "titulo": "Arma detectada por cámara",
            "descripcion": "Incidente en análisis — se actualizará con datos del infractor.",
            "tipo": "weapon_detection",
            "severidad": "alta",
            "estado": "abierta",
            "camara_id": CAMERA_ID,
            "evento_id": event_id,
        }
        if tienda_id is not None:
            alert_payload["tienda_id"] = tienda_id

        al = requests.post(f"{API_BASE}/alerts", json=alert_payload, headers=headers, timeout=10)
        al.raise_for_status()
        alert_id = al.json().get("id")
        result[0] = (event_id, alert_id)
        LOG.info("ALERTA #%s lanzada — evento #%s (cámara %d)", alert_id, event_id, CAMERA_ID)

    _with_token_refresh(token_ref, _post)
    return result[0]


def patch_weapon_result(
    token_ref: List[str],
    event_id: Optional[int],
    alert_id: Optional[int],
    confidence: float,
    identity: Optional[Dict],
) -> None:
    """PATCH event + alert after buffer window closes with identity info."""
    if identity and identity.get("known"):
        nombre          = identity.get("nombre") or ""
        ap_paterno      = identity.get("apellido_paterno") or ""
        ap_materno      = identity.get("apellido_materno") or ""
        full_parts      = [p for p in [nombre, ap_paterno, ap_materno] if p]
        full            = " ".join(full_parts)
        dni             = identity["dni"]
        event_comment = (
            f"Arma detectada — confianza: {confidence:.0%} | "
            f"Portador identificado: {full} (DNI: {dni}) | "
            f"Confianza rostro: {identity['face_score']:.0%}"
        )
        alert_desc = (
            f"Arma detectada en cámara {CAMERA_ID} | "
            f"Portador: {full} (DNI: {dni}) | "
            f"Confianza arma: {confidence:.0%}"
        )
    else:
        event_comment = (
            f"Arma detectada — confianza: {confidence:.0%} | "
            f"Lo sentimos, no se pudo identificar al infractor"
        )
        alert_desc = (
            f"Arma detectada en cámara {CAMERA_ID} — confianza: {confidence:.0%} | "
            f"Lo sentimos, no se pudo identificar al infractor"
        )

    def _patch(tok: str) -> None:
        h = {"Authorization": f"Bearer {tok}"}
        if event_id:
            requests.patch(
                f"{API_BASE}/events/{event_id}",
                json={"comentario": event_comment},
                headers=h, timeout=10,
            ).raise_for_status()
            # Save identification record directly linked to the event
            if identity and identity.get("known"):
                requests.post(
                    f"{API_BASE}/identifications",
                    json={
                        "evento_id":              event_id,
                        "nombre":                 identity.get("nombre"),
                        "apellido":               identity.get("apellido_paterno"),
                        "apellido_materno":       identity.get("apellido_materno"),
                        "dni":                    identity["dni"],
                        "edad":                   identity.get("edad"),
                        "confianza_identificacion": round(identity["face_score"], 4),
                        "fuente":                 "ia",
                    },
                    headers=h, timeout=10,
                )
        if alert_id:
            requests.patch(
                f"{API_BASE}/alerts/{alert_id}",
                json={"descripcion": alert_desc},
                headers=h, timeout=10,
            ).raise_for_status()
        LOG.info("Evento #%s y alerta #%s actualizados — identidad: %s", event_id, alert_id,
                 identity.get("nombre") if identity and identity.get("known") else "DESCONOCIDO")

    _with_token_refresh(token_ref, _patch)


# ── Detection helpers (sin cambios) ──────────────────────────────────────────

def get_arma_score(raw_score: float) -> Tuple[str, float]:
    arma_score = 1.0 - raw_score
    predicted_label = "arma" if arma_score >= NASNET_THRESHOLD else "no_arma"
    return predicted_label, arma_score


def predict_nasnet_crop(crop_bgr: np.ndarray, nasnet_model: tf.keras.Model) -> Tuple[str, float, float]:
    enhanced  = enhance_crop(crop_bgr)
    crop_rgb  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    batch     = np.expand_dims(resized.astype("float32"), axis=0)
    raw_score = float(nasnet_model(batch, training=False)[0][0])
    predicted_label, arma_score = get_arma_score(raw_score)
    return predicted_label, arma_score, raw_score


def extract_crop(frame_bgr: np.ndarray, box_xyxy: Sequence[float]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


def box_center(box_xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def euclidean_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return float(math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def boxes_intersect(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w   = max(0.0, inter_x2 - inter_x1)
    inter_h   = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a   = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b   = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def detection_priority(detection: Dict[str, object]) -> Tuple[int, float, float]:
    confirmed_rank     = 1 if detection.get("weapon_confirmed") else 0
    weapon_confidence  = float(detection.get("weapon_confidence") or 0.0)
    yolo_score         = float(detection.get("yolo_score") or 0.0)
    return confirmed_rank, weapon_confidence, yolo_score


def remove_duplicate_detections(
    detections: Sequence[Dict[str, object]],
    iou_threshold: float = IOU_DUPLICATE_THRESHOLD,
) -> List[Dict[str, object]]:
    ordered  = sorted(detections, key=detection_priority, reverse=True)
    selected: List[Dict[str, object]] = []
    for detection in ordered:
        box = detection.get("weapon_box")
        if box is None:
            selected.append(detection)
            continue
        duplicated = False
        for chosen in selected:
            chosen_box = chosen.get("weapon_box")
            if chosen_box is None:
                continue
            if calculate_iou(box, chosen_box) >= iou_threshold:
                duplicated = True
                break
        if not duplicated:
            selected.append(detection)
    return selected


def build_arm_zone(
    points: Sequence[Tuple[float, float]], margin: int = ARM_ZONE_MARGIN
) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def extract_visible_pose_points(
    result_pose, kp_conf_min: float = KP_CONF_MIN
) -> Dict[str, List[Tuple[float, float]]]:
    visible_points: Dict[str, List[Tuple[float, float]]] = {
        "wrists_elbows": [],
        "arm_zones": [],
    }
    if result_pose.keypoints is None or result_pose.keypoints.data is None:
        return visible_points

    keypoints_data = result_pose.keypoints.data.cpu().numpy()
    for person_points in keypoints_data:
        person_visible: Dict[int, Tuple[float, float]] = {}
        for kp_idx in (
            KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
            KP_LEFT_ELBOW,    KP_RIGHT_ELBOW,
            KP_LEFT_WRIST,    KP_RIGHT_WRIST,
        ):
            x, y, conf = person_points[kp_idx]
            if conf >= kp_conf_min:
                person_visible[kp_idx] = (float(x), float(y))

        for kp_idx in (KP_LEFT_ELBOW, KP_RIGHT_ELBOW, KP_LEFT_WRIST, KP_RIGHT_WRIST):
            if kp_idx in person_visible:
                visible_points["wrists_elbows"].append(person_visible[kp_idx])

        left_chain  = [kp for kp in (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW,  KP_LEFT_WRIST)  if kp in person_visible]
        right_chain = [kp for kp in (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST) if kp in person_visible]
        if len(left_chain) >= 2:
            visible_points["arm_zones"].append(build_arm_zone([person_visible[kp] for kp in left_chain]))
        if len(right_chain) >= 2:
            visible_points["arm_zones"].append(build_arm_zone([person_visible[kp] for kp in right_chain]))

    return visible_points


def decide_hand_proximity(
    gun_box: Sequence[float],
    visible_pose_points: Dict[str, List[Tuple[float, float]]],
) -> Tuple[bool, float]:
    center    = box_center(gun_box)
    keypoints = visible_pose_points["wrists_elbows"]
    arm_zones = visible_pose_points["arm_zones"]
    min_distance = float("inf")
    if keypoints:
        min_distance = min(euclidean_distance(center, kp) for kp in keypoints)
    near_keypoint      = min_distance <= HAND_DISTANCE_THRESHOLD
    intersects_arm_zone = any(boxes_intersect(gun_box, zone) for zone in arm_zones)
    return near_keypoint or intersects_arm_zone, min_distance


def draw_status(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def _save_snapshot(frame: np.ndarray, camera_id: int) -> None:
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(SNAPSHOT_DIR / f"cam{camera_id}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75],
        )
    except Exception:
        pass


def draw_confirmed_detection(frame: np.ndarray, detection: Dict[str, object]) -> None:
    weapon_box = detection.get("weapon_box")
    if weapon_box is None:
        return
    x1, y1, x2, y2 = [int(v) for v in weapon_box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)


# ── Frame quality & preprocessing ────────────────────────────────────────────

def frame_sharpness(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Used to skip blurry buffer frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_crop(crop: np.ndarray) -> np.ndarray:
    """CLAHE contrast + unsharp-mask on weapon crop before NASNet."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE_CROP.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)


def enhance_face_crop(crop: np.ndarray) -> np.ndarray:
    """Softer CLAHE + unsharp-mask on face crop before InsightFace."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE_FACE.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)


# ── Frame capture thread ──────────────────────────────────────────────────────

class LatestFrameCapture:
    """Background thread that reads RTSP as fast as possible.
    Detection loop picks up only the most recent decoded frame."""

    def __init__(self, url: str) -> None:
        self._url   = url
        self._frame: Optional[np.ndarray] = None
        self._ok    = False
        self._lock  = threading.Lock()
        self._stop  = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self) -> None:
        cap = self._open()
        _fail_count = 0
        _first_frame = True
        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                _fail_count += 1
                if _fail_count == 1:
                    LOG.error("Sin frame en %s — reconectando... (este mensaje se repite cada 30 intentos)", self._url)
                elif _fail_count % 30 == 0:
                    LOG.warning("Aún sin frame — %d intentos fallidos. URL: %s", _fail_count, self._url)
                cap.release()
                time.sleep(2)
                cap = self._open()
                continue
            if _first_frame:
                LOG.info("PRIMER FRAME recibido de %s — stream activo", self._url)
                _first_frame = False
            _fail_count = 0
            with self._lock:
                self._ok    = True
                self._frame = frame

        cap.release()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def release(self) -> None:
        self._stop = True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    paths = resolve_model_paths()
    validate_model_paths(paths)

    LOG.info("=" * 60)
    LOG.info("VISORA WEAPON WORKER — cámara %d", CAMERA_ID)
    LOG.info("RTSP URL: %s", RTSP_URL)
    LOG.info("API: %s  usuario: %s", API_BASE, API_USER)
    LOG.info("ALERT_COOLDOWN: %ds  FRAME_SKIP: %d  HEADLESS: %s", ALERT_COOLDOWN, FRAME_SKIP, HEADLESS)
    LOG.info("=" * 60)
    LOG.info("Cargando modelos YOLO + NASNet...")
    yolo_weapon  = YOLO(str(paths["yolo_weapon"]))
    yolo_pose    = YOLO(str(paths["yolo_pose"]))
    nasnet_model = tf.keras.models.load_model(
        str(paths["nasnet"]),
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )
    LOG.info("Modelos de detección listos")

    # ── InsightFace carga en hilo de fondo para no bloquear la detección ─────
    _face_app_ref: List[Optional[FaceAnalysis]] = [None]
    _face_ready   = threading.Event()
    centroids     = load_centroids(CENTROIDS_PATH)

    def _load_face_bg() -> None:
        try:
            LOG.info("Cargando InsightFace buffalo_l en segundo plano...")
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE), det_thresh=FACE_DET_THRESH)
            _face_app_ref[0] = app
            LOG.info("InsightFace listo — identificación facial activada")
        except Exception as exc:
            LOG.error("InsightFace no pudo cargar: %s", exc)
        finally:
            _face_ready.set()

    threading.Thread(target=_load_face_bg, daemon=True).start()

    LOG.info("Conectando al backend %s ...", API_BASE)
    try:
        token_ref = [api_login()]
    except Exception as exc:
        LOG.critical("FALLO DE LOGIN — worker no puede continuar: %s", exc)
        return
    tienda_id = get_tienda_id(token_ref[0])
    LOG.info("Cámara %d → tienda_id=%s", CAMERA_ID, tienda_id)

    LOG.info("Abriendo stream RTSP: %s", RTSP_URL)
    LOG.info("Si ves 'DESCRIBE failed 404' abajo, el stream no existe en MediaMTX — verifica que la fuente de video esté publicando en ese path.")
    cap = LatestFrameCapture(RTSP_URL)
    # wait for first frame (up to 10s) instead of fixed sleep
    _t0 = time.time()
    while not cap.read()[0] and time.time() - _t0 < 10:
        time.sleep(0.1)

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("Dispositivo inferencia: %s", _device)
    LOG.info("Detección ACTIVA (InsightFace cargando en fondo...)")

    last_alert_time = 0.0
    window_name = "VISORA — Weapon Detection"
    _frame_counter = 0
    _frames_processed = 0
    _last_stats_log = time.time()
    # fallback frame when buffer is empty (all frames filtered as blurry)
    _last_confirmed_frame: Optional[np.ndarray] = None
    _last_confirmed_conf:  float = 0.0
    _last_confirmed_wbox:  List[int] = []

    # ── Frame buffer (approach 2: best weapon-conf frame in 5s window) ─────────
    # Each entry: (frame, weapon_conf, sharpness, weapon_box)
    _buf: List[Tuple[np.ndarray, float, float, List[int]]] = []
    _buf_start        = 0.0
    _buffering        = False
    _pending_event_id: Optional[int] = None
    _pending_alert_id: Optional[int] = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            time.sleep(0.03)
            continue

        _frame_counter += 1
        _frames_processed += 1
        # Log stats every 60s so we know the worker is alive and processing
        if time.time() - _last_stats_log >= 60:
            LOG.info("ALIVE — frames procesados: %d | cooldown restante: %.0fs | buffering: %s",
                     _frames_processed, max(0, ALERT_COOLDOWN - (time.time() - last_alert_time)), _buffering)
            _last_stats_log = time.time()
        if _frame_counter % FRAME_SKIP != 0:
            if not HEADLESS:
                cv2.imshow(window_name, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        weapon_results = yolo_weapon.predict(
            source=frame_bgr, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False, device=_device,
        )
        result_weapon = weapon_results[0]

        # Run pose only when YOLO found at least one weapon candidate — skips ~90% of frames
        if result_weapon.boxes is not None and len(result_weapon.boxes) > 0:
            pose_results = yolo_pose.predict(
                source=frame_bgr, conf=POSE_CONF, imgsz=POSE_IMGSZ, verbose=False, device=_device,
            )
            result_pose = pose_results[0]
        else:
            result_pose = None

        visible_pose_points = (
            extract_visible_pose_points(result_pose, kp_conf_min=KP_CONF_MIN)
            if result_pose is not None
            else {"wrists_elbows": [], "arm_zones": []}
        )

        detections: List[Dict[str, object]] = []
        if result_weapon.boxes is not None:
            for box in result_weapon.boxes:
                box_xyxy   = box.xyxy[0].cpu().numpy().tolist()
                yolo_score = float(box.conf[0].item())
                near_hand_arm, min_distance = decide_hand_proximity(box_xyxy, visible_pose_points)

                detection: Dict[str, object] = {
                    "weapon_detected":  True,
                    "weapon_confirmed": False,
                    "weapon_confidence": None,
                    "yolo_score":       yolo_score,
                    "nasnet_score":     None,
                    "near_hand_arm":    near_hand_arm,
                    "weapon_box":       [int(round(v)) for v in box_xyxy],
                    "carrier_region":   None,
                    "min_distance":     None if not np.isfinite(min_distance) else float(min_distance),
                    "raw_score":        None,
                    "predicted_label":  None,
                    "reason":           "yolo_candidate",
                }

                if visible_pose_points["arm_zones"]:
                    detection["carrier_region"] = [int(round(v)) for v in visible_pose_points["arm_zones"][0]]

                if not near_hand_arm:
                    detection["reason"] = "outside_hand_arm_zone"
                else:
                    crop = extract_crop(frame_bgr, box_xyxy)
                    if crop is None or crop.size == 0:
                        detection["reason"] = "empty_crop"
                    else:
                        predicted_label, arma_score, raw_score = predict_nasnet_crop(crop, nasnet_model)
                        detection["nasnet_score"]      = arma_score
                        detection["weapon_confidence"] = arma_score
                        detection["raw_score"]         = raw_score
                        detection["predicted_label"]   = predicted_label
                        if predicted_label == "arma" and arma_score >= NASNET_THRESHOLD:
                            detection["weapon_confirmed"] = True
                            detection["reason"]           = "confirmed_by_nasnet"
                            LOG.debug("YOLO candidato CONFIRMADO por NASNet — yolo=%.2f nasnet=%.2f", yolo_score, arma_score)
                        else:
                            detection["reason"] = "rejected_by_nasnet"
                            LOG.debug("YOLO candidato RECHAZADO por NASNet — yolo=%.2f nasnet=%.2f (umbral=%.2f)", yolo_score, arma_score, NASNET_THRESHOLD)

                detections.append(detection)

        detections = remove_duplicate_detections(detections, IOU_DUPLICATE_THRESHOLD)
        confirmed_detections = remove_duplicate_detections(
            [d for d in detections if d.get("weapon_confirmed")],
            IOU_DUPLICATE_THRESHOLD,
        )

        if confirmed_detections:
            best       = confirmed_detections[0]
            confidence = float(best.get("weapon_confidence") or 0.0)
            weapon_box = list(best.get("weapon_box") or [0, 0, 0, 0])

            if PAUSE_FILE.exists():
                if _buffering:
                    _buffering = False
                    _buf.clear()
            else:
                now       = time.time()
                sharpness = frame_sharpness(frame_bgr)

                # Always keep the latest confirmed frame as fallback
                _last_confirmed_frame = frame_bgr.copy()
                _last_confirmed_conf  = confidence
                _last_confirmed_wbox  = weapon_box

                # Start buffer window when cooldown allows — fire alert immediately
                if not _buffering and now - last_alert_time >= ALERT_COOLDOWN:
                    _buffering = True
                    _buf_start = now
                    _buf.clear()
                    _pending_event_id, _pending_alert_id = post_weapon_initial(token_ref, tienda_id)
                    LOG.info("ARMA CONFIRMADA — confianza=%.0f%% | ventana de captura %.0fs iniciada | evento=#%s alerta=#%s",
                             confidence * 100, CAPTURE_WINDOW, _pending_event_id, _pending_alert_id)

                # Collect confirmed frames — accept frames above a soft blur threshold
                if _buffering and sharpness >= BLUR_MIN_VAR:
                    _buf.append((frame_bgr.copy(), confidence, sharpness, weapon_box))

                # Process buffer when window closes — update alert/event with identity + best frame
                if _buffering and now - _buf_start >= CAPTURE_WINDOW:
                    _buffering = False
                    # Use buffer frames if available; fall back to last confirmed frame
                    if _buf:
                        max_conf  = max(e[1] for e in _buf) or 1.0
                        max_sharp = max(e[2] for e in _buf) or 1.0
                        ranked = sorted(
                            _buf,
                            key=lambda e: 0.7 * (e[1] / max_conf) + 0.3 * (e[2] / max_sharp),
                            reverse=True,
                        )
                        top3 = ranked[:3]
                        best_frame, best_conf, _, best_wbox = top3[0]
                    elif _last_confirmed_frame is not None:
                        print("[VISORA] Buffer vacío — usando último frame confirmado")
                        best_frame = _last_confirmed_frame
                        best_conf  = _last_confirmed_conf
                        best_wbox  = _last_confirmed_wbox
                        top3 = [(best_frame, best_conf, 0.0, best_wbox)]
                    else:
                        print("[VISORA] Sin frames disponibles — solo alerta sin imagen")
                        patch_weapon_result(token_ref, _pending_event_id, _pending_alert_id, 0.0, None)
                        last_alert_time = time.time()
                        _buf.clear()
                        _pending_event_id = None
                        _pending_alert_id = None
                        continue

                    # Try to identify carrier across top-3 frames (stop on first known match)
                    identity: Optional[Dict] = None
                    face_app = _face_app_ref[0]
                    if face_app is None:
                        print("[VISORA] InsightFace aún cargando — identificación omitida")
                    else:
                        for cand_frame, _, _, cand_wbox in top3:
                            pose_cand = yolo_pose.predict(
                                source=cand_frame, conf=POSE_CONF, imgsz=POSE_IMGSZ,
                                verbose=False, device=_device,
                            )[0]
                            identity = identify_carrier(
                                cand_frame, cand_wbox, face_app, centroids, pose_cand,
                            )
                            if identity and identity.get("known"):
                                break

                    if identity:
                        tag = identity.get("nombre", "DESCONOCIDO") if identity.get("known") else "DESCONOCIDO"
                        LOG.info("Portador: %s (face_score=%.2f, known=%s)", tag, identity.get("face_score", 0), identity.get("known"))
                    else:
                        LOG.info("No se pudo identificar al portador — sin rostro detectado")

                    # Save best frame as snapshot (served by /detect/snapshot endpoint)
                    snap = best_frame.copy()
                    draw_confirmed_detection(snap, {"weapon_box": best_wbox})
                    _save_snapshot(snap, CAMERA_ID)

                    # PATCH event + alert with identity result
                    patch_weapon_result(token_ref, _pending_event_id, _pending_alert_id, best_conf, identity)
                    last_alert_time = time.time()

                    _buf.clear()
                    _pending_event_id = None
                    _pending_alert_id = None

            if not HEADLESS:
                draw_confirmed_detection(frame_bgr, best)
                lbl = f"ARMA — {'CAPTURANDO' if _buffering else 'CONFIRMADA'}"
                draw_status(frame_bgr, lbl, (0, 60, 255) if _buffering else (0, 0, 255))

        elif detections:
            LOG.debug("YOLO detectó %d candidato(s) pero todos rechazados por NASNet/pose", len(detections))
            if not HEADLESS:
                draw_status(frame_bgr, "Candidato rechazado", (0, 255, 255))
        else:
            if not HEADLESS:
                draw_status(frame_bgr, "Sin arma detectada", (0, 255, 0))

        if not HEADLESS:
            cv2.imshow(window_name, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    LOG.info("Worker finalizado — cámara %d", CAMERA_ID)
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
