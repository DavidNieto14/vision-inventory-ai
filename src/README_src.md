# src/

Código fuente del sistema de visión por computadora para inventario automático.

## Módulos

| Archivo           | Responsabilidad                                                    |
|-------------------|--------------------------------------------------------------------|
| `detector.py`     | Carga el modelo YOLOv8 y ejecuta inferencia sobre imágenes/video  |
| `counter.py`      | Agrega detecciones y mantiene conteos acumulados por categoría     |
| `database.py`     | Persistencia de detecciones y sesiones en SQLite                   |
| `preprocessor.py` | Preprocesamiento de imágenes (resize, CLAHE) y extracción de frames|
| `augmentation.py` | Generación de dataset sintético con Albumentations                 |

## Flujo de datos

```
VideoFrameExtractor → ImagePreprocessor → PieceDetector → PieceCounter
                                                       ↘
                                                  InventoryDatabase → CSV export
```

## Dependencias principales

- `ultralytics` — YOLOv8
- `opencv-python` — captura de video y procesamiento de imagen
- `albumentations` — augmentation de datos
- `pandas` — manipulación de resultados
- `python-dotenv` — gestión de configuración via `.env`
