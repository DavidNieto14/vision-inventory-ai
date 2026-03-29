FROM python:3.10-slim

# Metadatos
LABEL maintainer="DavidNieto14"
LABEL description="Sistema de visión por computadora para inventario en línea de pintura"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias del sistema (OpenCV headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python primero (capa de caché)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código fuente
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY .env.example .env

# Crear directorios de datos
RUN mkdir -p data/raw data/synthetic data/annotated data/exports

EXPOSE 8000

CMD ["python", "src/detector.py"]
