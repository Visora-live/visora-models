from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import tensorflow as tf
from tensorflow.keras.applications.nasnet import preprocess_input


YOLO_CONF = 0.25
YOLO_IMGSZ = 960
POSE_CONF = 0.35
POSE_IMGSZ = 640
NASNET_THRESHOLD = 0.5
HAND_DISTANCE_THRESHOLD = 120
KP_CONF_MIN = 0.25
ARM_ZONE_MARGIN = 40

SHOW_REJECTED = False
SHOW_POSE_DEBUG = False
SHOW_ARM_ZONES = False

KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_class_names(train_dir: Path) -> List[str]:
    class_names = sorted([path.name for path in train_dir.iterdir() if path.is_dir()])
    if len(class_names) != 2:
        raise ValueError(f"Se esperaban 2 clases binarias y se encontraron: {class_names}")
    return class_names


def get_arma_score(raw_score: float, class_names: Sequence[str], nasnet_threshold: float) -> Tuple[str, float]:
    if list(class_names) == ["arma", "no_arma"]:
        arma_score = 1.0 - raw_score
    elif list(class_names) == ["no_arma", "arma"]:
        arma_score = raw_score
    else:
        raise ValueError(f"Orden de clases no soportado para binario: {class_names}")

    predicted_label = "arma" if arma_score >= nasnet_threshold else "no_arma"
    return predicted_label, arma_score


def predict_nasnet_crop(
    crop_bgr: np.ndarray,
    nasnet_model: tf.keras.Model,
    class_names: Sequence[str],
    nasnet_threshold: float,
) -> Tuple[str, float, float]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    batch = np.expand_dims(resized.astype("float32"), axis=0)
    raw_score = float(nasnet_model.predict(batch, verbose=0)[0][0])
    predicted_label, arma_score = get_arma_score(raw_score, class_names, nasnet_threshold)
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


def build_arm_zone(points: Sequence[Tuple[float, float]], margin: int = ARM_ZONE_MARGIN) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def extract_visible_pose_points(result_pose, kp_conf_min: float = KP_CONF_MIN) -> Dict[str, List[Tuple[float, float]]]:
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
            KP_LEFT_SHOULDER,
            KP_RIGHT_SHOULDER,
            KP_LEFT_ELBOW,
            KP_RIGHT_ELBOW,
            KP_LEFT_WRIST,
            KP_RIGHT_WRIST,
        ):
            x, y, conf = person_points[kp_idx]
            if conf >= kp_conf_min:
                person_visible[kp_idx] = (float(x), float(y))

        for kp_idx in (KP_LEFT_ELBOW, KP_RIGHT_ELBOW, KP_LEFT_WRIST, KP_RIGHT_WRIST):
            if kp_idx in person_visible:
                visible_points["wrists_elbows"].append(person_visible[kp_idx])

        left_chain = [kp for kp in (KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST) if kp in person_visible]
        right_chain = [kp for kp in (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST) if kp in person_visible]

        if len(left_chain) >= 2:
            visible_points["arm_zones"].append(build_arm_zone([person_visible[kp] for kp in left_chain]))
        if len(right_chain) >= 2:
            visible_points["arm_zones"].append(build_arm_zone([person_visible[kp] for kp in right_chain]))

    return visible_points


def draw_pose_debug(frame_bgr: np.ndarray, result_pose, kp_conf_min: float = KP_CONF_MIN) -> None:
    if result_pose.keypoints is None or result_pose.keypoints.data is None:
        return

    keypoints_data = result_pose.keypoints.data.cpu().numpy()
    debug_indices = [
        KP_LEFT_SHOULDER,
        KP_RIGHT_SHOULDER,
        KP_LEFT_ELBOW,
        KP_RIGHT_ELBOW,
        KP_LEFT_WRIST,
        KP_RIGHT_WRIST,
    ]

    for person_points in keypoints_data:
        for kp_idx in debug_indices:
            x, y, conf = person_points[kp_idx]
            if conf >= kp_conf_min:
                cv2.circle(frame_bgr, (int(x), int(y)), 4, (255, 255, 0), -1)
                cv2.putText(
                    frame_bgr,
                    str(kp_idx),
                    (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )


def draw_arm_zones(frame_bgr: np.ndarray, arm_zones: Iterable[Tuple[float, float, float, float]]) -> None:
    for zone in arm_zones:
        x1, y1, x2, y2 = [int(v) for v in zone]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 180, 0), 1)


def decide_hand_proximity(
    gun_box: Sequence[float],
    visible_pose_points: Dict[str, List[Tuple[float, float]]],
    hand_distance_threshold: float,
) -> Tuple[bool, float]:
    center = box_center(gun_box)
    keypoints = visible_pose_points["wrists_elbows"]
    arm_zones = visible_pose_points["arm_zones"]

    min_distance = float("inf")
    if keypoints:
        min_distance = min(euclidean_distance(center, kp) for kp in keypoints)

    near_keypoint = min_distance <= hand_distance_threshold
    intersects_arm_zone = any(boxes_intersect(gun_box, zone) for zone in arm_zones)
    accepted = near_keypoint or intersects_arm_zone
    return accepted, min_distance


def draw_detection(
    frame_bgr: np.ndarray,
    box_xyxy: Sequence[float],
    text: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame_bgr,
        text,
        (x1, max(y1 - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    default_output = project_root() / "workspace_modelos" / "reports" / "webcam_yolo_pose_nasnet" / "webcam_yolo_pose_nasnet_fase4.mp4"
    parser = argparse.ArgumentParser(description="Webcam pipeline: YOLO Pose + YOLO Arma + NASNet Fase 4")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--show-pose-debug", action="store_true")
    parser.add_argument("--show-arm-zones", action="store_true")
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONF)
    parser.add_argument("--nasnet-threshold", type=float, default=NASNET_THRESHOLD)
    parser.add_argument("--hand-distance-threshold", type=float, default=HAND_DISTANCE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    yolo_weapon_path = root / "workspace_modelos" / "content-detector" / "weights" / "best.pt"
    yolo_pose_path = root / "yolov8n-pose.pt"
    nasnet_model_path = root / "workspace_modelos" / "models" / "nasnetmobile_weapon_validator_fase4_finetune_50k_final.keras"
    class_train_dir = root / "workspace_modelos" / "dataset_clasificacion_nasnet_v2_50k" / "train"
    output_video_path = args.output

    if not args.no_save:
        ensure_output_dir(output_video_path)

    for required_path, label in (
        (yolo_weapon_path, "modelo YOLO arma"),
        (yolo_pose_path, "modelo YOLO pose"),
        (nasnet_model_path, "modelo NASNetMobile"),
        (class_train_dir, "train de clasificacion"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"No se encontro el {label}: {required_path}")

    class_names = load_class_names(class_train_dir)
    print("class_names detectadas:", class_names)

    yolo_weapon = YOLO(str(yolo_weapon_path))
    yolo_pose = YOLO(str(yolo_pose_path))
    nasnet_model = tf.keras.models.load_model(
        str(nasnet_model_path),
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara con index {args.camera_index}")

    writer = None
    if not args.no_save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"No se pudo crear el video de salida: {output_video_path}")

    stats = {
        "frames_procesados": 0,
        "detecciones_yolo_totales": 0,
        "detecciones_cerca_mano_brazo": 0,
        "detecciones_confirmadas_nasnet": 0,
        "detecciones_rechazadas_nasnet": 0,
        "detecciones_fuera_zona_mano_brazo": 0,
    }

    window_name = "YOLO Pose + YOLO Arma + NASNet Fase 4"

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("No se pudo leer frame desde la camara.")
            break

        stats["frames_procesados"] += 1

        weapon_results = yolo_weapon.predict(
            source=frame_bgr,
            conf=args.yolo_conf,
            imgsz=YOLO_IMGSZ,
            verbose=False,
            device="cpu",
        )
        pose_results = yolo_pose.predict(
            source=frame_bgr,
            conf=POSE_CONF,
            imgsz=POSE_IMGSZ,
            verbose=False,
            device="cpu",
        )

        result_weapon = weapon_results[0]
        result_pose = pose_results[0]
        visible_pose_points = extract_visible_pose_points(result_pose, kp_conf_min=KP_CONF_MIN)

        if args.show_pose_debug or SHOW_POSE_DEBUG:
            draw_pose_debug(frame_bgr, result_pose, kp_conf_min=KP_CONF_MIN)
        if args.show_arm_zones or SHOW_ARM_ZONES:
            draw_arm_zones(frame_bgr, visible_pose_points["arm_zones"])

        if result_weapon.boxes is not None:
            for box in result_weapon.boxes:
                stats["detecciones_yolo_totales"] += 1
                box_xyxy = box.xyxy[0].cpu().numpy().tolist()
                yolo_score = float(box.conf[0].item())

                is_near_hand, min_distance = decide_hand_proximity(
                    box_xyxy,
                    visible_pose_points,
                    hand_distance_threshold=args.hand_distance_threshold,
                )

                if not is_near_hand:
                    stats["detecciones_fuera_zona_mano_brazo"] += 1
                    if args.show_rejected or SHOW_REJECTED:
                        draw_detection(
                            frame_bgr,
                            box_xyxy,
                            f"fuera zona mano | yolo={yolo_score:.2f}",
                            color=(140, 140, 140),
                            thickness=1,
                        )
                    continue

                stats["detecciones_cerca_mano_brazo"] += 1
                crop = extract_crop(frame_bgr, box_xyxy)
                if crop is None or crop.size == 0:
                    stats["detecciones_rechazadas_nasnet"] += 1
                    if args.show_rejected or SHOW_REJECTED:
                        draw_detection(
                            frame_bgr,
                            box_xyxy,
                            "rechazado | crop invalido",
                            color=(0, 255, 255),
                            thickness=2,
                        )
                    continue

                predicted_label, arma_score, raw_score = predict_nasnet_crop(
                    crop,
                    nasnet_model,
                    class_names,
                    nasnet_threshold=args.nasnet_threshold,
                )

                if predicted_label == "arma" and arma_score >= args.nasnet_threshold:
                    stats["detecciones_confirmadas_nasnet"] += 1
                    draw_detection(
                        frame_bgr,
                        box_xyxy,
                        f"ARMA CONFIRMADA | yolo={yolo_score:.2f} | arma={arma_score:.2f}",
                        color=(0, 0, 255),
                        thickness=2,
                    )
                else:
                    stats["detecciones_rechazadas_nasnet"] += 1
                    if args.show_rejected or SHOW_REJECTED:
                        draw_detection(
                            frame_bgr,
                            box_xyxy,
                            f"rechazado | yolo={yolo_score:.2f} | arma={arma_score:.2f} | raw={raw_score:.2f} | d={min_distance:.1f}",
                            color=(0, 255, 255),
                            thickness=2,
                        )

        cv2.imshow(window_name, frame_bgr)
        if writer is not None:
            writer.write(frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\nResumen final:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    if writer is not None:
        print(f"- video_salida: {output_video_path}")


if __name__ == "__main__":
    main()
