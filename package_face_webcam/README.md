# Demo local - Reconocimiento facial con camara web

## Descripcion

Este paquete permite probar localmente el reconocimiento facial con camara web usando una galeria personalizada y centroides ya generados.

La demo usa realmente:

- `InsightFace`
- galeria personalizada
- centroides

No usa backend, FastAPI, base de datos, eventos ni deteccion de armas.

## Contenido del paquete

Estructura real incluida:

- `run_face_webcam.py`
- `requirements.txt`
- `README.md`
- `run_windows.ps1`
- `run_linux.sh`
- `custom_gallery/`
- `data/custom_identities.csv`
- `reports/custom_face_enrollment/custom_gallery_centroids.csv`

## Modelos y datos incluidos

Archivos y datos realmente presentes en el paquete:

- `custom_gallery/`
- `data/custom_identities.csv`
- `reports/custom_face_enrollment/custom_gallery_centroids.csv`

## Configuracion usada

Constantes reales leidas desde `run_face_webcam.py`:

- `CAMERA_INDEX = 0`
- `CTX_ID = -1`
- `DET_SIZE = 320`
- `DET_THRESH = 0.2`
- `PAD_RATIO = 0.35`
- `THRESHOLD = 0.35`
- `WINDOW_NAME = "Face Recognition Webcam"`
- `CENTROIDS_PATH = PROJECT_ROOT / "reports" / "custom_face_enrollment" / "custom_gallery_centroids.csv"`

Configuracion efectiva:

- usa CPU por defecto porque `CTX_ID = -1`
- carga centroides desde `reports/custom_face_enrollment/custom_gallery_centroids.csv`
- no carga una ruta separada de galeria en tiempo de ejecucion; compara directamente contra centroides
- no carga un modelo YOLO facial dentro de este paquete

## Flujo de funcionamiento de la demo

1. Abre la camara web con OpenCV.
2. Detecta rostro con InsightFace.
3. Genera embedding con InsightFace.
4. Compara el embedding contra los centroides cargados al inicio.
5. Muestra resultados en pantalla sobre el video en vivo.

## Instalacion en Windows

```powershell
py -3.11 -m venv .venv_face
.\.venv_face\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si quieres usar el script de arranque incluido:

```powershell
.\run_windows.ps1
```

## Instalacion en Linux

```bash
python3 -m venv .venv_face
source .venv_face/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si quieres usar el script de arranque incluido:

```bash
bash run_linux.sh
```

## Ejecucion de la demo

```powershell
python run_face_webcam.py
```

## Cambio de camara

Si la camara no abre, cambia:

- `CAMERA_INDEX = 0`

por:

- `CAMERA_INDEX = 1`
- `CAMERA_INDEX = 2`

dentro de `run_face_webcam.py`.

## Salir de la demo

La demo sale al presionar:

- `Q`

## Que se vera en pantalla

Textos reales usados por el script:

- `Identificado: ... | score=...`
- `Desconocido | score=...`
- `Rostro no detectado`

## Problemas comunes

- la camara no abre: cambia `CAMERA_INDEX = 0` por `1` o `2`
- faltan dependencias: instala `requirements.txt`
- centroides no encontrados: verifica `reports/custom_face_enrollment/custom_gallery_centroids.csv`
- `InsightFace` puede descargar modelos al primer uso si no estan en cache local
- el uso de CPU puede ser mas lento porque el script usa `CTX_ID = -1`

## Advertencias

- no existe `reports/custom_face_enrollment/custom_gallery_embeddings.csv` dentro de este paquete
- no existe `models/yolo_face.pt` dentro de este paquete
- el paquete depende de que `custom_gallery_centroids.csv` ya exista y sea valido
