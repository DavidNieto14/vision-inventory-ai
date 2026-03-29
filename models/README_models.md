# models/

Este directorio almacena los pesos del modelo YOLOv8 entrenado.

## Archivos esperados

| Archivo      | Descripción                                                    |
|--------------|----------------------------------------------------------------|
| `best.pt`    | Mejor checkpoint durante el entrenamiento (mayor mAP en val)  |
| `last.pt`    | Último checkpoint del entrenamiento                           |
| `yolov8n.pt` | Pesos preentrenados base de Ultralytics (COCO) — descargados automáticamente |

## Cómo obtener `best.pt`

1. Ejecutar el notebook `notebooks/02_entrenamiento.ipynb`.
2. Al finalizar el entrenamiento, copiar el archivo desde `runs/train/expX/weights/best.pt` a este directorio:

```bash
cp runs/train/exp1/weights/best.pt models/best.pt
```

## Notas

- Los archivos `.pt` **no se suben al repositorio** (están en `.gitignore`) por su tamaño.
- Para compartir pesos entrenados usa releases de GitHub o almacenamiento externo (Google Drive, S3).
