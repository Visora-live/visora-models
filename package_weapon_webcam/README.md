# Demo local - Detección de armas con cámara web

## Descripción

Este paquete permite probar localmente una demo de detección de armas usando una cámara web. La demo abre la cámara con OpenCV, procesa los frames en tiempo real y muestra en pantalla si hay un arma confirmada o si un candidato fue rechazado.

El paquete incluye únicamente estos componentes del pipeline:

- YOLO detector de armas
- YOLO-Pose
- NASNetMobile Fase 4

No incluye reconocimiento facial, backend, endpoints, base de datos, flujo de eventos ni integración con otros módulos.

## Contenido del paquete

Estructura real incluida:

- `run_weapon_webcam.py`
- `requirements.txt`
- `README.md`
- `run_windows.ps1`
- `run_linux.sh`
- `yolov8n-pose.pt`
- `workspace_modelos/content-detector/weights/best.pt`
- `workspace_modelos/models/nasnetmobile_weapon_validator_fase4_finetune_50k_final.keras`

## Modelos incluidos

Modelos reales encontrados dentro del paquete:

- `workspace_modelos/content-detector/weights/best.pt`
- `yolov8n-pose.pt`
- `workspace_modelos/models/nasnetmobile_weapon_validator_fase4_finetune_50k_final.keras`

## Configuración usada

Constantes reales definidas en `run_weapon_webcam.py`:

- `CAMERA_INDEX = 0`
- `YOLO_CONF = 0.25`
- `YOLO_IMGSZ = 960`
- `POSE_CONF = 0.35`
- `POSE_IMGSZ = 640`
- `NASNET_THRESHOLD = 0.5`
- `HAND_DISTANCE_THRESHOLD = 120.0`
- `KP_CONF_MIN = 0.25`
- `ARM_ZONE_MARGIN = 40`
- `DEBUG = True`
- `IOU_DUPLICATE_THRESHOLD = 0.5`

Uso de CPU / GPU:

- El script fuerza uso de CPU:
  - `os.environ["CUDA_VISIBLE_DEVICES"] = ""`
- En las predicciones YOLO usa:
  - `device="cpu"`

## Flujo de funcionamiento de la demo

1. Abre la cámara web con OpenCV.
2. Captura frames en tiempo real.
3. Ejecuta YOLO detector de armas sobre el frame.
4. Ejecuta YOLO-Pose para obtener puntos y zonas mano/brazo.
5. Filtra candidatos por cercanía/intersección con mano/brazo.
6. Valida el recorte con NASNetMobile.
7. Muestra el resultado en pantalla.

## Instalación en Windows

Si existe `requirements.txt`, instalar así:

```powershell
py -3.11 -m venv .venv_weapon
.\.venv_weapon\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si existe `run_windows.ps1`, puede ejecutarse así:

```powershell
.\run_windows.ps1
```

## Instalación en Linux

El paquete incluye `run_linux.sh`, por lo tanto también aplica instalación Linux:

```bash
python3 -m venv .venv_weapon
source .venv_weapon/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si se desea usar el script incluido:

```bash
bash run_linux.sh
```

## Ejecución de la demo

El script principal real del paquete es:

```powershell
python run_weapon_webcam.py
```

## Cambio de cámara

El script define:

- `CAMERA_INDEX = 0`

Si la cámara no abre, cambiar dentro de `run_weapon_webcam.py`:

- `CAMERA_INDEX = 0`
- por `1`
- o por `2`

## Salir de la demo

La demo se cierra presionando:

- `Q`

## Qué se verá en pantalla

Textos reales encontrados en el código:

- `Sin arma detectada`
- `Candidato rechazado`
- `ARMA CONFIRMADA`

Además, cuando hay un arma confirmada se dibuja la caja y se muestra la confianza:

- `ARMA CONFIRMADA | arma=0.xx`

## Problemas comunes

- La cámara no abre:
  - cambiar `CAMERA_INDEX = 0` por `1` o `2`
- Dependencias faltantes:
  - reinstalar con `python -m pip install -r requirements.txt`
- Modelo no encontrado:
  - verificar que existan los tres modelos incluidos en el paquete
- Versión de Python incorrecta:
  - usar Python `3.11` recomendado
- Uso de CPU puede ser lento:
  - el script está configurado para CPU por defecto

## Advertencias

- No se detectaron archivos de modelo faltantes dentro de `package_weapon_webcam/`.
- El paquete está orientado solo a prueba local con cámara web.
- No incluye backend, API, base de datos, eventos, reconocimiento facial ni entrenamiento.
