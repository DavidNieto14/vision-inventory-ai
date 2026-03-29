# data/

Directorio de datos del proyecto. Contiene imágenes originales, sintéticas, anotadas y resultados exportados.

## Estructura

```
data/
├── raw/           # Imágenes originales capturadas en la línea de pintura (sin procesar)
├── synthetic/     # Imágenes generadas por augmentation (src/augmentation.py)
│   ├── images/
│   └── labels/
├── annotated/     # Imágenes con anotaciones YOLO listas para entrenamiento
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── exports/       # CSVs de resultados de detección por sesión
```

## Formato de anotaciones

Las etiquetas siguen el formato YOLO:

```
<class_id> <cx> <cy> <width> <height>
```

Donde `cx`, `cy`, `width` y `height` están normalizados en el rango `[0, 1]`.

## Notas

- Las imágenes crudas y sintéticas **no se suben al repositorio** (ver `.gitignore`).
- Los exports CSV sí pueden versionarse si son pequeños.
- La base de datos SQLite (`inventory.db`) se genera en tiempo de ejecución y tampoco se versiona.
