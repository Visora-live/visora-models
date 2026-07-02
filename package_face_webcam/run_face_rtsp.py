from __future__ import annotations

import csv
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0")

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ── Config via env vars ───────────────────────────────────────────────────────
CAMERA_ID        = int(os.getenv("CAMERA_ID", "1"))
RTSP_URL         = os.getenv("RTSP_URL", f"rtsp://localhost:8554/cam{CAMERA_ID}")
HEADLESS         = os.getenv("HEADLESS", "0") == "1"
FRAME_SKIP       = int(os.getenv("FRAME_SKIP", "1"))
SNAPSHOT_DIR     = Path(os.getenv("SNAPSHOT_DIR", r"C:\visora_snapshots"))
PAUSE_FILE       = SNAPSHOT_DIR / f"cam{CAMERA_ID}.face_paused"

# ── Identificación bajo pedido (disparada por el worker de armas) ─────────────
# El worker de armas escribe cam{ID}_evt{event_id}.json cuando detecta un arma;
# este worker responde con cam{ID}_evt{event_id}.result.json. Ya no genera
# eventos/alertas propios — solo identifica cuando se le pide.
PENDING_IDENT_DIR = Path(os.getenv("PENDING_IDENT_DIR", "/opt/visora/workers/shared/pending_ident"))

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


# ── Identificación bajo pedido ────────────────────────────────────────────────

def _closest_face(faces: List, weapon_box: Optional[Sequence[float]]):
    if not faces:
        return None
    if not weapon_box:
        return faces[0]
    wx = (weapon_box[0] + weapon_box[2]) / 2.0
    wy = (weapon_box[1] + weapon_box[3]) / 2.0
    best_face = None
    best_dist = None
    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox.astype(float)
        fx, fy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        dist = ((fx - wx) ** 2 + (fy - wy) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_dist, best_face = dist, face
    return best_face


def respond_pending_identifications(faces: List, centroids: List[Dict]) -> None:
    """Answers pending identification requests dropped by the weapon worker.

    This worker no longer creates its own events/alerts — it only reacts to
    a weapon detection, identifying whichever face is closest to the weapon.
    """
    if not faces:
        return
    for req_path in sorted(PENDING_IDENT_DIR.glob(f"cam{CAMERA_ID}_evt*.json")):
        if req_path.name.endswith(".result.json"):
            continue
        try:
            req = json.loads(req_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            LOG.error("Pedido de identificación ilegible (%s): %s", req_path.name, exc)
            req_path.unlink(missing_ok=True)
            continue

        face = _closest_face(faces, req.get("weapon_box"))
        if face is None:
            continue

        emb = (
            face.normed_embedding
            if hasattr(face, "normed_embedding") and face.normed_embedding is not None
            else face.embedding
        )
        best, score = match_face(emb, centroids)
        is_known = best is not None and score >= THRESHOLD

        result = {
            "known": is_known,
            "face_score": round(float(score), 4),
            "nombre": best.get("nombre") if is_known else None,
            "apellido_paterno": best.get("apellido_paterno") if is_known else None,
            "apellido_materno": best.get("apellido_materno") if is_known else None,
            "dni": best.get("dni") if is_known else None,
            "edad": best.get("edad") if is_known else None,
        }
        res_path = req_path.parent / (req_path.stem + ".result.json")
        try:
            res_path.write_text(json.dumps(result))
        except OSError as exc:
            LOG.error("No se pudo escribir resultado de identificación: %s", exc)
        req_path.unlink(missing_ok=True)
        LOG.info(
            "Identificación respondida — evento #%s: %s (score=%.2f)",
            req.get("evento_id"), result["nombre"] if is_known else "DESCONOCIDO", score,
        )


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
    LOG.info("THRESHOLD: %.2f  FRAME_SKIP: %d", THRESHOLD, FRAME_SKIP)
    LOG.info("Identificación bajo pedido — pending dir: %s", PENDING_IDENT_DIR)
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

    PENDING_IDENT_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("Abriendo stream RTSP: %s", RTSP_URL)
    LOG.info("Si ves 'DESCRIBE failed 404', el stream no existe en MediaMTX — verifica que la fuente esté publicando.")
    cap = LatestFrameCapture(RTSP_URL)
    time.sleep(2)

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
            LOG.info("ALIVE — frames procesados: %d", frames_processed)
            last_stats_log = time.time()

        if frame_counter % FRAME_SKIP != 0:
            if not HEADLESS:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        if PAUSE_FILE.exists():
            LOG.debug("Detección pausada (pause file existe)")
            if not HEADLESS:
                cv2.putText(frame, "Deteccion pausada", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
            save_snapshot(frame)
            respond_pending_identifications(faces, centroids)

            if not HEADLESS:
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

                    if is_known:
                        color = (0, 180, 0)
                        parts = [p for p in [best.get("nombre"), best.get("apellido_paterno"), best.get("apellido_materno")] if p]
                        full  = " ".join(parts) if parts else best["dni"]
                        label = f"{full} ({score:.2f})"
                    else:
                        color = (0, 60, 200)
                        label = f"Desconocido ({score:.2f})"

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
