# VISORA — Models

Workers de inteligencia artificial para el sistema de monitoreo de seguridad VISORA. Procesan video en tiempo real desde cámaras RTSP y reportan detecciones al backend vía API REST.

## Paquetes

### `package_weapon_webcam` — Detección de armas

Pipeline de detección de armas de fuego en tiempo real:

1. **YOLOv8** — detector de candidatos a arma
2. **YOLOv8-Pose** — extrae keypoints del cuerpo (solo cuando YOLO detecta candidatos)
3. **NASNetMobile** — valida el recorte del candidato descartando falsos positivos

Reporta eventos y alertas al backend con snapshot adjunto cuando se confirma un arma.

### `package_face_webcam` — Identificación facial

Pipeline de identificación de personas:

1. **InsightFace** — detección + generación de embeddings faciales
2. **Galería personalizada** — compara embeddings contra centroides de identidades enroladas

Reporta identificaciones al backend ligadas a eventos existentes.

## Requisitos de hardware

- GPU NVIDIA recomendada (RTX 3050 o superior) para inferencia en tiempo real
- CPU funcional pero más lento
- Python 3.11

## Configuración

Cada worker se configura mediante variables de entorno:

```env
CAMERA_ID=1
RTSP_URL=rtsp://localhost:8554/cam1
API_BASE=http://localhost:8000/api
VISORA_USER=admin
VISORA_PASS=tu_password
FRAME_SKIP=1
ALERT_COOLDOWN=30
CAPTURE_WINDOW=1.0
```

## Modelos (pesos)

Los pesos no están en el repositorio. Colócalos en sus rutas antes de ejecutar:

| Archivo | Ruta esperada |
|---------|--------------|
| `best.pt` (detector de armas) | `package_weapon_webcam/workspace_modelos/content-detector/weights/best.pt` |
| `yolov8n-pose.pt` | `package_weapon_webcam/yolov8n-pose.pt` |
| `nasnetmobile_weapon_validator_fase4_finetune_50k_final.keras` | `package_weapon_webcam/workspace_modelos/models/` |

InsightFace descarga sus modelos automáticamente al primer uso.

## Instalación

Ver el `README.md` de cada paquete para instrucciones detalladas de instalación en Windows y Linux.

## Arquitectura de integración

```
Cámara RTSP
    │
    ▼
Worker (este repo)
    │  POST /api/events
    │  POST /api/alerts
    │  POST /api/identifications
    ▼
visora-backend
    │
    ▼
visora-frontend
```
