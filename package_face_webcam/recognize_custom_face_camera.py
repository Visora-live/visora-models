"""
Reconocimiento facial en camara contra la galeria personalizada.

Usa centroides generados desde custom_gallery y no depende de frmdb ni de YOLO.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REQUIRED_DEPENDENCIES = {
    "insightface": "insightface",
    "onnxruntime": "onnxruntime",
    "opencv-python": "cv2",
    "numpy": "numpy",
}


def validate_dependencies_or_exit() -> None:
    """Valida dependencias antes de importar librerias pesadas."""
    missing = [
        package_name
        for package_name, import_name in REQUIRED_DEPENDENCIES.items()
        if importlib.util.find_spec(import_name) is None
    ]
    if missing:
        print("ERROR: faltan dependencias para reconocimiento por camara.", file=sys.stderr)
        print(f"Dependencias faltantes: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


validate_dependencies_or_exit()

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENTROIDS = PROJECT_ROOT / "reports" / "custom_face_enrollment" / "custom_gallery_centroids.csv"


def parse_args() -> argparse.Namespace:
    """Define argumentos CLI."""
    parser = argparse.ArgumentParser(description="Reconocimiento facial en vivo contra custom_gallery.")
    parser.add_argument(
        "--centroids",
        default="reports/custom_face_enrollment/custom_gallery_centroids.csv",
        help="CSV de centroides personalizados.",
    )
    parser.add_argument("--camera-index", default=0, type=int, help="Indice de camara para OpenCV.")
    parser.add_argument("--ctx-id", default=-1, type=int, help="-1 CPU; 0 GPU si esta disponible.")
    parser.add_argument("--det-size", default=320, type=int, help="Tamano de deteccion InsightFace.")
    parser.add_argument("--det-thresh", default=0.2, type=float, help="Umbral de deteccion InsightFace.")
    parser.add_argument("--pad-ratio", default=0.35, type=float, help="Padding por lado antes de detectar.")
    parser.add_argument(
        "--unknown-threshold",
        default=0.35,
        type=float,
        help="Si Top-1 cae debajo de este score, marca el rostro como DESCONOCIDO.",
    )
    parser.add_argument("--window-name", default="Custom Face Recognition", help="Nombre de ventana OpenCV.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    """Resuelve rutas relativas al proyecto."""
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_vector(vector_text: str) -> np.ndarray:
    """Parsea un vector serializado con separador |."""
    values = [float(value) for value in str(vector_text).split("|") if value.strip()]
    if not values:
        raise ValueError("Vector de centroide vacio.")
    return normalize_embedding(np.asarray(values, dtype=np.float32))


def load_centroids(path: Path) -> List[Dict[str, object]]:
    """Carga centroides personalizados desde CSV."""
    if not path.exists():
        raise FileNotFoundError(
            "No existe el archivo de centroides. Ejecuta primero enroll_custom_faces.py."
        )

    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"identity_id", "nombre", "dni", "centroid_vector"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas en centroides: {', '.join(sorted(missing))}")

        for row in reader:
            rows.append(
                {
                    "identity_id": str(row.get("identity_id", "")).strip(),
                    "nombre": str(row.get("nombre", "")).strip(),
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


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normaliza embedding L2."""
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Embedding con norma cero.")
    return embedding / norm


def embedding_from_face(face) -> np.ndarray:
    """Obtiene embedding normalizado desde InsightFace."""
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return normalize_embedding(face.normed_embedding)
    if hasattr(face, "embedding") and face.embedding is not None:
        return normalize_embedding(face.embedding)
    raise ValueError("InsightFace no devolvio embedding.")


def build_face_app(ctx_id: int, det_size: int, det_thresh: float) -> FaceAnalysis:
    """Inicializa InsightFace preentrenado."""
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
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


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
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
    """Ejecuta reconocimiento en vivo contra centroides personalizados."""
    capture = None
    try:
        args = parse_args()
        centroids_path = resolve_path(args.centroids)
        centroids = load_centroids(centroids_path)
        app = build_face_app(args.ctx_id, args.det_size, args.det_thresh)

        capture = cv2.VideoCapture(args.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara con indice {args.camera_index}.")

        print("=== Reconocimiento personalizado en camara ===")
        print("Galeria: custom_gallery -> centroides personalizados")
        print(f"Centroides: {centroids_path}")
        print("Presiona q para salir.")

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("No se pudo leer un frame de la camara.", file=sys.stderr)
                break

            frame = normalize_image(frame)
            padded_frame, pad_x, pad_y = apply_padding(frame, args.pad_ratio)
            faces = app.get(padded_frame)

            output = frame.copy()
            for face in faces:
                embedding = embedding_from_face(face)
                ranked = rank_centroids(embedding, centroids)
                if not ranked:
                    continue

                top1, top1_score = ranked[0]
                top5 = ranked[:5]
                is_known = top1_score >= args.unknown_threshold

                x1, y1, x2, y2 = face.bbox.astype(float)
                left, top, right, bottom = clamp_box(
                    x1 - pad_x,
                    y1 - pad_y,
                    x2 - pad_x,
                    y2 - pad_y,
                    output.shape[1],
                    output.shape[0],
                )

                color = (0, 180, 0) if is_known else (0, 90, 220)
                label_identity = top1["identity_id"] if is_known else "DESCONOCIDO"
                label_name = str(top1["nombre"]).strip() if is_known else ""
                label = f"{label_identity} {top1_score:.2f}".strip()
                if label_name:
                    label = f"{label_identity} | {label_name} | {top1_score:.2f}"

                cv2.rectangle(output, (left, top), (right, bottom), color, 2)
                draw_label(output, label, left, top, color)

                top5_text = ", ".join(
                    f"{row['identity_id']}:{score:.2f}" for row, score in top5
                )
                print(
                    f"Rostro detectado -> Top-1: {top1['identity_id']} | "
                    f"Nombre: {top1['nombre']} | DNI: {top1['dni']} | "
                    f"Score: {top1_score:.4f} | Top-5: {top5_text}"
                )

            if not faces:
                cv2.putText(
                    output,
                    "Sin rostro detectado",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(args.window_name, output)
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
    sys.exit(main())
