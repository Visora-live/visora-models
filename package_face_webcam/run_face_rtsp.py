from __future__ import annotations

import csv
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0")

import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis

# ── Config via env vars ───────────────────────────────────────────────────────
CAMERA_ID        = int(os.getenv("CAMERA_ID", "1"))
RTSP_URL         = os.getenv("RTSP_URL", f"rtsp://localhost:8554/cam{CAMERA_ID}")
API_BASE         = os.getenv("API_BASE", "http://localhost:8000/api")
API_USER         = os.getenv("VISORA_USER", "admin")
API_PASS         = os.getenv("VISORA_PASS", "")
HEADLESS         = os.getenv("HEADLESS", "0") == "1"
FRAME_SKIP       = int(os.getenv("FRAME_SKIP", "1"))
UNKNOWN_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "30"))
KNOWN_COOLDOWN   = int(os.getenv("KNOWN_COOLDOWN", "120"))
SNAPSHOT_DIR     = Path(os.getenv("SNAPSHOT_DIR", r"C:\visora_snapshots"))
PAUSE_FILE       = SNAPSHOT_DIR / f"cam{CAMERA_ID}.face_paused"

_DEFAULT_CENTROIDS = (
    Path(__file__).resolve().parent
    / "reports" / "custom_face_enrollment" / "custom_gallery_centroids.csv"
)
CENTROIDS_PATH = Path(os.getenv("CENTROIDS_PATH", str(_DEFAULT_CENTROIDS)))

# ── InsightFace config ────────────────────────────────────────────────────────
DET_SIZE    = 320
DET_THRESH  = 0.35
THRESHOLD   = float(os.getenv("FACE_THRESHOLD", "0.38"))
PAD_RATIO   = 0.25


# ── Structured logger ─────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SNAPSHOT_DIR / f"cam{CAMERA_ID}_face_diag.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger(f"visora.face.cam{CAMERA_ID}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

LOG = _setup_logger()


# ── Auth ──────────────────────────────────────────────────────────────────────

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
        LOG.warning("No se pudo obtener tienda_id: %s", exc)
    return None


# ── API reporting ─────────────────────────────────────────────────────────────

def _post_with_refresh(
    token_ref: List[str],
    method: str,
    url: str,
    payload: Dict,
) -> Optional[Dict]:
    def _attempt(tok: str) -> requests.Response:
        return requests.request(
            method, url,
            json=payload,
            headers={"Authorization": f"Bearer {tok}"},
            timeout=10,
        )

    try:
        resp = _attempt(token_ref[0])
        if resp.status_code in (401, 403):
            LOG.warning("Token expirado — renovando...")
            token_ref[0] = api_login()
            resp = _attempt(token_ref[0])
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        LOG.error("Error HTTP %s %s: %s", method, url, exc)
        return None


def post_known_face(
    token_ref: List[str],
    tienda_id: Optional[int],
    nombre: Optional[str],
    apellido_paterno: Optional[str],
    apellido_materno: Optional[str],
    dni: str,
    edad: Optional[int],
    score: float,
) -> None:
    parts = [p for p in [nombre, apellido_paterno, apellido_materno] if p]
    full_name = " ".join(parts) if parts else dni
    comment = f"Identificado: {full_name} (DNI: {dni}) — confianza {score:.2f}"

    result = _post_with_refresh(
        token_ref, "POST", f"{API_BASE}/events",
        {
            "tipo": "facial_recognition",
            "severidad": "baja",
            "estado": "abierto",
            "camara_id": CAMERA_ID,
            "comentario": comment,
        },
    )
    if result:
        event_id = result.get("id")
        LOG.info("ROSTRO CONOCIDO — evento #%s | %s (DNI:%s) score=%.2f", event_id, full_name, dni, score)
        if event_id:
            _post_with_refresh(
                token_ref, "POST", f"{API_BASE}/identifications",
                {
                    "evento_id":                event_id,
                    "nombre":                   nombre,
                    "apellido":                 apellido_paterno,
                    "apellido_materno":         apellido_materno,
                    "dni":                      dni,
                    "edad":                     edad,
                    "confianza_identificacion": round(score, 4),
                    "fuente":                   "ia",
                },
            )


def post_unknown_face(
    token_ref: List[str],
    tienda_id: Optional[int],
    best_score: float,
) -> None:
    comment = f"Rostro desconocido — similitud máxima: {best_score:.2f}"

    ev = _post_with_refresh(
        token_ref, "POST", f"{API_BASE}/events",
        {
            "tipo": "facial_recognition",
            "severidad": "media",
            "estado": "abierto",
            "camara_id": CAMERA_ID,
            "comentario": comment,
        },
    )
    event_id = ev.get("id") if ev else None

    alert_payload: Dict = {
        "titulo": "Rostro no identificado detectado",
        "descripcion": comment,
        "tipo": "facial_recognition",
        "severidad": "media",
        "estado": "abierta",
        "camara_id": CAMERA_ID,
    }
    if event_id:
        alert_payload["evento_id"] = event_id
    if tienda_id is not None:
        alert_payload["tienda_id"] = tienda_id

    result = _post_with_refresh(token_ref, "POST", f"{API_BASE}/alerts", alert_payload)
    if result:
        LOG.info("ROSTRO DESCONOCIDO — alerta #%s | score=%.2f", result.get("id"), best_score)


# ── Centroids ─────────────────────────────────────────────────────────────────

def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def load_centroids(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Centroides no encontrados: {path}")
    rows: List[Dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(v) for v in str(row["centroid_vector"]).split("|") if v.strip()]
            edad_raw = row.get("edad", "").strip()
            rows.append({
                "identity_id":      str(row.get("identity_id", "")).strip(),
                "nombre":           str(row.get("nombre", "")).strip() or None,
                "apellido_paterno": str(row.get("apellido_paterno", "")).strip() or None,
                "apellido_materno": str(row.get("apellido_materno", "")).strip() or None,
                "dni":              str(row.get("dni", "")).strip(),
                "edad":             int(edad_raw) if edad_raw.isdigit() else None,
                "centroid":         normalize_embedding(np.array(vals, dtype=np.float32)),
            })
    return rows


def match_face(embedding: np.ndarray, centroids: List[Dict]) -> Tuple[Optional[Dict], float]:
    if not centroids:
        return None, 0.0
    emb = normalize_embedding(embedding)
    scores = [(c, float(np.dot(emb, c["centroid"]))) for c in centroids]
    best, best_score = max(scores, key=lambda x: x[1])
    return best, best_score


# ── Snapshot ──────────────────────────────────────────────────────────────────

def save_snapshot(frame: np.ndarray) -> None:
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(SNAPSHOT_DIR / f"cam{CAMERA_ID}_face.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75],
        )
    except Exception:
        pass


# ── RTSP capture thread ───────────────────────────────────────────────────────

class LatestFrameCapture:
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
        cap.set(cv2.CAP_PROP_FPS, 25)
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
                    LOG.error("Sin frame en %s — reconectando... (mensaje cada 30 intentos)", self._url)
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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def release(self) -> None:
        self._stop = True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    LOG.info("=" * 60)
    LOG.info("VISORA FACE WORKER — cámara %d", CAMERA_ID)
    LOG.info("RTSP URL: %s", RTSP_URL)
    LOG.info("API: %s  usuario: %s", API_BASE, API_USER)
    LOG.info("THRESHOLD: %.2f  KNOWN_COOLDOWN: %ds  UNKNOWN_COOLDOWN: %ds  FRAME_SKIP: %d",
             THRESHOLD, KNOWN_COOLDOWN, UNKNOWN_COOLDOWN, FRAME_SKIP)
    LOG.info("=" * 60)

    LOG.info("Cargando centroides desde %s ...", CENTROIDS_PATH)
    try:
        centroids = load_centroids(CENTROIDS_PATH)
        LOG.info("%d identidades cargadas", len(centroids))
    except FileNotFoundError as exc:
        LOG.critical("No se encontraron centroides — worker no puede continuar: %s", exc)
        return

    LOG.info("Inicializando InsightFace buffalo_l...")
    face_app = FaceAnalysis(name="buffalo_l")
    try:
        import onnxruntime as ort
        _providers = ort.get_available_providers()
        _ctx = 0 if "CUDAExecutionProvider" in _providers else -1
    except Exception:
        _ctx = -1
    face_app.prepare(ctx_id=_ctx, det_size=(DET_SIZE, DET_SIZE), det_thresh=DET_THRESH)
    LOG.info("InsightFace listo — ctx_id=%d (%s)", _ctx, "GPU" if _ctx == 0 else "CPU")

    LOG.info("Conectando al backend %s ...", API_BASE)
    try:
        token_ref = [api_login()]
    except Exception as exc:
        LOG.critical("FALLO DE LOGIN — worker no puede continuar: %s", exc)
        return
    tienda_id = get_tienda_id(token_ref[0])
    LOG.info("Cámara %d → tienda_id=%s", CAMERA_ID, tienda_id)

    LOG.info("Abriendo stream RTSP: %s", RTSP_URL)
    LOG.info("Si ves 'DESCRIBE failed 404', el stream no existe en MediaMTX — verifica que la fuente esté publicando.")
    cap = LatestFrameCapture(RTSP_URL)
    time.sleep(2)

    last_known:   Dict[str, float] = {}
    last_unknown  = 0.0
    frame_counter = 0
    frames_processed = 0
    last_stats_log = time.time()
    window_name   = "VISORA — Face Detection"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.1)
            continue

        frame_counter += 1
        frames_processed += 1

        if time.time() - last_stats_log >= 60:
            LOG.info("ALIVE — frames procesados: %d | cooldown unknown restante: %.0fs",
                     frames_processed, max(0, UNKNOWN_COOLDOWN - (time.time() - last_unknown)))
            last_stats_log = time.time()

        if frame_counter % FRAME_SKIP != 0:
            if not HEADLESS:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        h, w = frame.shape[:2]
        pad_x = int(w * PAD_RATIO)
        pad_y = int(h * PAD_RATIO)
        padded = cv2.copyMakeBorder(
            frame, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
        )

        faces = face_app.get(padded)
        output = frame.copy()

        if not faces:
            if not HEADLESS:
                cv2.putText(output, "Sin rostro", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)
        else:
            LOG.debug("%d rostro(s) detectado(s) en frame", len(faces))
            for face in faces:
                emb = (
                    face.normed_embedding
                    if hasattr(face, "normed_embedding") and face.normed_embedding is not None
                    else face.embedding
                )
                best, score = match_face(emb, centroids)
                is_known = best is not None and score >= THRESHOLD

                x1, y1, x2, y2 = face.bbox.astype(float)
                left  = max(0, int(x1 - pad_x))
                top   = max(0, int(y1 - pad_y))
                right = min(w - 1, int(x2 - pad_x))
                bot   = min(h - 1, int(y2 - pad_y))

                if PAUSE_FILE.exists():
                    LOG.debug("Detección pausada (pause file existe)")
                    if not HEADLESS:
                        cv2.putText(output, "Deteccion pausada", (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    break

                if is_known:
                    color = (0, 180, 0)
                    parts = [p for p in [best.get("nombre"), best.get("apellido_paterno"), best.get("apellido_materno")] if p]
                    full  = " ".join(parts) if parts else best["dni"]
                    label = f"{full} ({score:.2f})"
                    now   = time.time()
                    iid   = best["identity_id"]
                    cooldown_left = KNOWN_COOLDOWN - (now - last_known.get(iid, 0.0))
                    if cooldown_left <= 0:
                        LOG.debug("Rostro conocido match: %s score=%.2f — registrando evento", full, score)
                        post_known_face(
                            token_ref, tienda_id,
                            best.get("nombre"), best.get("apellido_paterno"),
                            best.get("apellido_materno"), best["dni"],
                            best.get("edad"), score,
                        )
                        last_known[iid] = now
                        save_snapshot(frame)
                    else:
                        LOG.debug("Rostro conocido %s score=%.2f — cooldown %.0fs restante", full, score, cooldown_left)
                else:
                    color = (0, 60, 200)
                    label = f"Desconocido ({score:.2f})"
                    now   = time.time()
                    cooldown_left = UNKNOWN_COOLDOWN - (now - last_unknown)
                    if cooldown_left <= 0:
                        LOG.debug("Rostro desconocido score=%.2f — registrando evento+alerta", score)
                        post_unknown_face(token_ref, tienda_id, score)
                        last_unknown = now
                        save_snapshot(frame)
                    else:
                        LOG.debug("Rostro desconocido score=%.2f — cooldown %.0fs restante", score, cooldown_left)

                if not HEADLESS:
                    cv2.rectangle(output, (left, top), (right, bot), color, 2)
                    cv2.putText(output, label, (left, max(0, top - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        if not HEADLESS:
            cv2.imshow(window_name, output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    LOG.info("Worker facial finalizado — cámara %d", CAMERA_ID)
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
