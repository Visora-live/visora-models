"""
Demo local de reconocimiento facial por webcam.

Este script:
- abre la camara local con OpenCV
- detecta rostro con InsightFace
- extrae embedding
- compara contra centroides personalizados
- muestra el resultado en vivo
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parent
CAMERA_INDEX = 0
CTX_ID = -1
DET_SIZE = 320
DET_THRESH = 0.2
PAD_RATIO = 0.35
THRESHOLD = 0.35
WINDOW_NAME = "Face Recognition Webcam"
CENTROIDS_PATH = PROJECT_ROOT / "reports" / "custom_face_enrollment" / "custom_gallery_centroids.csv"


def split_name(full_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Divide un nombre completo usando una heuristica simple."""
    tokens = [token for token in full_name.split() if token]
    if not tokens:
        return None, None
    if len(tokens) == 1:
        return tokens[0], None
    return tokens[0], " ".join(tokens[1:])


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normaliza embedding L2."""
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Embedding con norma cero.")
    return embedding / norm


def parse_vector(vector_text: str) -> np.ndarray:
    """Parsea un vector serializado con separador |."""
    values = [float(value) for value in str(vector_text).split("|") if value.strip()]
    if not values:
        raise ValueError("Centroide vacio encontrado en el CSV.")
    return normalize_embedding(np.asarray(values, dtype=np.float32))


def load_centroids(path: Path) -> List[Dict[str, object]]:
    """Carga centroides personalizados desde CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Asegura que el paquete incluya custom_gallery_centroids.csv."
        )

    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"identity_id", "nombre", "dni", "centroid_vector"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas en centroides: {', '.join(sorted(missing))}")

        for row in reader:
            full_name = str(row.get("nombre", "")).strip()
            nombre, apellido = split_name(full_name)
            rows.append(
                {
                    "identity_id": str(row.get("identity_id", "")).strip(),
                    "full_name": full_name,
                    "nombre": nombre,
                    "apellido": apellido,
                    "dni": str(row.get("dni", "")).strip(),
                    "centroid": parse_vector(row.get("centroid_vector", "")),
                }
            )

    if not rows:
        raise ValueError("El archivo de centroides no contiene identidades.")
    return rows


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normaliza imagen a BGR uint8."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        if np.max(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def apply_padding(image: np.ndarray, pad_ratio: float) -> Tuple[np.ndarray, int, int]:
    """Agrega borde replicado y devuelve offsets para reproyectar cajas."""
    if pad_ratio <= 0:
        return image, 0, 0

    height, width = image.shape[:2]
    pad_x = int(round(width * pad_ratio))
    pad_y = int(round(height * pad_ratio))
    if pad_x <= 0 and pad_y <= 0:
        return image, 0, 0

    padded = cv2.copyMakeBorder(
        image,
        top=pad_y,
        bottom=pad_y,
        left=pad_x,
        right=pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )
    return padded, pad_x, pad_y


def face_area(face) -> float:
    """Calcula area del rostro detectado."""
    x1, y1, x2, y2 = face.bbox.astype(float)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def embedding_from_face(face) -> np.ndarray:
    """Obtiene embedding normalizado desde InsightFace."""
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return normalize_embedding(face.normed_embedding)
    if hasattr(face, "embedding") and face.embedding is not None:
        return normalize_embedding(face.embedding)
    raise ValueError("InsightFace no devolvio embedding.")


def build_face_app() -> FaceAnalysis:
    """Inicializa InsightFace preentrenado."""
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=CTX_ID, det_size=(DET_SIZE, DET_SIZE), det_thresh=DET_THRESH)
    return app


def rank_centroids(
    embedding: np.ndarray, centroids: List[Dict[str, object]]
) -> List[Tuple[Dict[str, object], float]]:
    """Calcula ranking por similitud coseno."""
    ranked = []
    for row in centroids:
        score = float(np.dot(embedding, row["centroid"]))
        ranked.append((row, score))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def clamp_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    """Ajusta bounding box a limites validos de imagen."""
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width - 1, int(round(x2))))
    bottom = max(0, min(height - 1, int(round(y2))))
    return left, top, right, bottom


def draw_label(frame: np.ndarray, text: str, left: int, top: int, color: Tuple[int, int, int]) -> None:
    """Dibuja etiqueta compacta sobre la imagen."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y1 = max(0, top - text_height - baseline - 6)
    y2 = max(text_height + baseline + 6, top)
    x2 = left + text_width + 10
    cv2.rectangle(frame, (left, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (left + 5, y2 - baseline - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> int:
    """Ejecuta la demo de webcam."""
    capture = None
    try:
        centroids = load_centroids(CENTROIDS_PATH)
        app = build_face_app()

        capture = cv2.VideoCapture(CAMERA_INDEX)
        if not capture.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la camara con indice {CAMERA_INDEX}. Cambia CAMERA_INDEX a 1 o 2."
            )

        print("=== Face Webcam Demo ===")
        print(f"Centroides cargados: {len(centroids)}")
        print(f"Threshold: {THRESHOLD}")
        print("Presiona Q para salir.")

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("No se pudo leer un frame de la camara.", file=sys.stderr)
                break

            frame = normalize_image(frame)
            padded_frame, pad_x, pad_y = apply_padding(frame, PAD_RATIO)
            faces = app.get(padded_frame)
            output = frame.copy()

            if not faces:
                cv2.putText(
                    output,
                    "Rostro no detectado",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                print("Rostro no detectado")

            for face in faces:
                embedding = embedding_from_face(face)
                ranked = rank_centroids(embedding, centroids)
                if not ranked:
                    continue

                top1, top1_score = ranked[0]
                is_known = top1_score >= THRESHOLD

                x1, y1, x2, y2 = face.bbox.astype(float)
                left, top, right, bottom = clamp_box(
                    x1 - pad_x,
                    y1 - pad_y,
                    x2 - pad_x,
                    y2 - pad_y,
                    output.shape[1],
                    output.shape[0],
                )

                if is_known:
                    color = (0, 180, 0)
                    surname = top1["apellido"] or ""
                    full_label_name = f"{top1['nombre'] or ''} {surname}".strip()
                    label = f"Identificado: {full_label_name} | score={top1_score:.2f}"
                    print(
                        f"Identificado: {full_label_name} | dni={top1['dni']} | score={top1_score:.4f}"
                    )
                else:
                    color = (0, 90, 220)
                    label = f"Desconocido | score={top1_score:.2f}"
                    print(f"Desconocido | score={top1_score:.4f}")

                cv2.rectangle(output, (left, top), (right, bottom), color, 2)
                draw_label(output, label, left, top, color)

            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
