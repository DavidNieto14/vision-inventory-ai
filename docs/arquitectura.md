# Arquitectura del Sistema

## Descripción General

El sistema de visión por computadora para inventario automático procesa video en tiempo real proveniente de cámaras ubicadas en la línea de pintura industrial. Detecta y cuenta piezas por categoría usando YOLOv8, persiste los resultados en SQLite y los exporta como CSV.

---

## Diagrama de Componentes

```
┌──────────────────────────────────────────────────────────────────┐
│                         Fuente de Video                          │
│              (Cámara IP / USB / Archivo .mp4)                    │
└─────────────────────────────┬────────────────────────────────────┘
                              │ frames BGR
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     VideoFrameExtractor                          │
│                     (src/preprocessor.py)                        │
│   - Captura frames a fps_target                                  │
│   - Entrega (frame_id, frame_BGR) como generador                 │
└─────────────────────────────┬────────────────────────────────────┘
                              │ frame_BGR
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     ImagePreprocessor                            │
│                     (src/preprocessor.py)                        │
│   - Letterbox resize a 640×640                                   │
│   - CLAHE para mejora de contraste                               │
└─────────────────────────────┬────────────────────────────────────┘
                              │ frame preprocesado
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      PieceDetector                               │
│                      (src/detector.py)                           │
│   - Modelo: YOLOv8 (Ultralytics)                                 │
│   - Salida: [{class_name, confidence, bbox}, ...]                │
└──────────────┬──────────────────────────────┬────────────────────┘
               │ detections                   │ detections
               ▼                              ▼
┌──────────────────────────┐    ┌─────────────────────────────────┐
│      PieceCounter        │    │       InventoryDatabase          │
│      (src/counter.py)    │    │       (src/database.py)          │
│  - Conteo por clase      │    │  - Inserta detección por frame   │
│  - Historial por frame   │    │  - Gestiona sesiones             │
│  - Export CSV            │    │  - Queries de resumen            │
└──────────────────────────┘    └─────────────────────────────────┘
```

---

## Módulos

### `src/detector.py` — PieceDetector
- Carga el modelo `.pt` entrenado con Ultralytics YOLO.
- Ejecuta inferencia por frame o en lote.
- Filtra por umbral de confianza e IoU (NMS).

### `src/counter.py` — PieceCounter
- Recibe la lista de detecciones de cada frame.
- Mantiene un acumulado de conteos por clase durante la sesión.
- Genera DataFrame con historial y lo exporta a CSV.

### `src/database.py` — InventoryDatabase
- Esquema SQLite con tablas `detections` y `sessions`.
- Inserta registros atómicamente usando context manager.
- Provee queries para auditoría y reportes.

### `src/preprocessor.py` — ImagePreprocessor / VideoFrameExtractor
- Normaliza imágenes al formato esperado por YOLOv8.
- Controla la tasa de extracción de frames para no saturar la inferencia.

### `src/augmentation.py` — SyntheticAugmentor
- Amplía el dataset de entrenamiento con transformaciones Albumentations.
- Adapta los bounding boxes YOLO tras cada transformación geométrica.

---

## Stack Tecnológico

| Capa            | Tecnología             | Versión mínima |
|-----------------|------------------------|----------------|
| Detección       | YOLOv8 (Ultralytics)   | 8.0.0          |
| Framework ML    | PyTorch                | 2.0.0          |
| Visión          | OpenCV                 | 4.8.0          |
| Augmentación    | Albumentations         | 1.3.0          |
| Persistencia    | SQLite (built-in)      | —              |
| Contenedores    | Docker / Compose       | 24.x           |
| Python          | CPython                | 3.10+          |

---

## Decisiones de Diseño

1. **YOLOv8n como arquitectura base**: balance entre velocidad de inferencia y precisión, adecuado para hardware industrial sin GPU dedicada.
2. **SQLite en lugar de PostgreSQL**: el sistema opera en modo standalone; SQLite elimina dependencias de servidor y simplifica el despliegue.
3. **Albumentations para augmentation**: integración nativa con formato YOLO para adaptar bounding boxes automáticamente.
4. **Docker para despliegue**: garantiza reproducibilidad en el entorno de producción industrial.

---

## Métricas Objetivo

| Métrica          | Objetivo |
|------------------|----------|
| mAP@50           | ≥ 0.85   |
| Reducción discrepancia inventario | ≥ 30%  |
| Latencia por frame (CPU) | < 100 ms |
