# Sistema de Visión por Computadora para Inventario Automático en Línea de Pintura

Sistema de detección y conteo automático de piezas en una línea de pintura industrial, basado en YOLOv8 y procesamiento de video en tiempo real. El objetivo es reducir en al menos 30% la discrepancia entre inventario físico y digital mediante visión por computadora.

Repositorio: [https://github.com/DavidNieto14/vision-inventory-ai](https://github.com/DavidNieto14/vision-inventory-ai)

---

## Stack Tecnológico

| Componente       | Tecnología                  |
|------------------|-----------------------------|
| Detección        | YOLOv8 (Ultralytics)        |
| Framework ML     | PyTorch 2.x                 |
| Visión           | OpenCV 4.x                  |
| Persistencia     | SQLite                      |
| Contenedores     | Docker / Docker Compose     |
| Notebooks        | Jupyter                     |
| Augmentación     | Albumentations              |
| Testing          | Pytest                      |

---

## Instalación

### Requisitos previos
- Python 3.10+
- Docker (opcional)
- GPU con CUDA 11.8+ (recomendado)

### Instalación local

```bash
# Clonar el repositorio
git clone https://github.com/DavidNieto14/vision-inventory-ai.git
cd vision-inventory-ai

# Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus valores
```

### Instalación con Docker

```bash
docker-compose up --build
```

---

## Uso Básico

### Detección en imagen

```python
from src.detector import PieceDetector

detector = PieceDetector(model_path="models/best.pt")
results = detector.detect("data/raw/imagen.jpg")
print(results)
```

### Detección en video / cámara

```python
from src.detector import PieceDetector

detector = PieceDetector(model_path="models/best.pt")
detector.run_video(source=0)  # 0 = webcam, o ruta a archivo de video
```

### Conteo por categoría

```python
from src.counter import PieceCounter

counter = PieceCounter()
totals = counter.count_from_results(results)
print(totals)
```

---

## Estructura del Repositorio

```
vision-inventory-ai/
├── configs/                    # Archivos de configuración YAML
│   ├── model_config.yaml
│   ├── db_config.yaml
│   └── experiment_config.yaml
├── data/
│   ├── raw/                    # Imágenes originales sin procesar
│   ├── synthetic/              # Imágenes sintéticas generadas
│   ├── annotated/              # Imágenes con anotaciones YOLO
│   └── exports/                # CSVs de resultados de detección
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── docs/
│   ├── arquitectura.md         # Descripción técnica del sistema
│   └── README_docs.md
├── models/                     # Pesos entrenados (.pt)
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_entrenamiento.ipynb
│   └── 03_validacion.ipynb
├── src/
│   ├── detector.py             # Módulo principal de detección YOLOv8
│   ├── counter.py              # Lógica de conteo por categoría
│   ├── database.py             # Módulo SQLite para persistencia
│   ├── preprocessor.py         # Preprocesamiento de imágenes y video
│   └── augmentation.py         # Data augmentation sintética
├── tests/
│   ├── test_detector.py
│   ├── test_counter.py
│   └── test_database.py
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── README.md
└── requirements.txt
```

---

## Metodología

El proyecto se desarrolla bajo marco **Scrum** con sprints de dos semanas, siguiendo el flujo:

1. Recolección y anotación de imágenes reales de la línea de pintura
2. Generación de imágenes sintéticas con augmentation
3. Entrenamiento y validación del modelo YOLOv8
4. Integración con base de datos SQLite para trazabilidad
5. Despliegue en contenedor Docker

---

## Licencia

Proyecto académico — Maestría en Innovación en AI.
