"""
preprocessor.py
---------------
Preprocesamiento de video e imágenes para el pipeline de detección.

Responsabilidades:
- Cargar y validar archivos de video.
- Extraer metadatos del video (fps, resolución, duración).
- Extraer frames a una tasa configurable y guardarlos en disco.
- Redimensionar frames al tamaño requerido por el modelo.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class VideoPreprocessor:
    """
    Preprocesador de video para extracción y preparación de frames.

    Proporciona utilidades para cargar videos, obtener sus metadatos,
    extraer frames a intervalos regulares y redimensionar imágenes
    al formato requerido por el modelo de detección.
    """

    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """
        Valida la existencia del archivo de video y retorna un objeto VideoCapture.

        Args:
            video_path: Ruta al archivo de video (.mp4, .avi, etc.).

        Returns:
            Objeto cv2.VideoCapture listo para leer frames.

        Raises:
            FileNotFoundError: Si el archivo no existe en la ruta indicada.
            RuntimeError: Si OpenCV no puede abrir el archivo de video.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo de video no encontrado: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV no pudo abrir el video: {video_path}")

        return cap

    def get_video_info(self, cap: cv2.VideoCapture) -> Dict:
        """
        Extrae los metadatos principales de un objeto VideoCapture.

        Args:
            cap: Objeto cv2.VideoCapture abierto y válido.

        Returns:
            Diccionario con las claves:
                - 'fps'              (float): Cuadros por segundo del video.
                - 'total_frames'     (int): Número total de frames.
                - 'width'            (int): Ancho del frame en píxeles.
                - 'height'           (int): Alto del frame en píxeles.
                - 'duration_seconds' (float): Duración total en segundos.
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps if fps > 0 else 0.0

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": round(duration_seconds, 2),
        }

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        sample_rate: int = 1,
    ) -> int:
        """
        Extrae frames del video a una tasa de muestreo y los guarda como imágenes JPG.

        Por defecto extrae 1 frame por segundo. Los archivos se nombran con el
        patrón frame_XXXXXX.jpg usando el número de frame como identificador.

        Args:
            video_path: Ruta al archivo de video de entrada.
            output_dir: Directorio donde se guardarán los frames extraídos.
            sample_rate: Número de frames entre cada extracción. Con sample_rate=1
                         se extrae 1 frame por segundo (basado en el FPS del video).

        Returns:
            Número total de frames guardados en disco.

        Raises:
            FileNotFoundError: Si el archivo de video no existe.
            RuntimeError: Si el video no puede abrirse.
        """
        cap = self.load_video(video_path)
        info = self.get_video_info(cap)
        fps = info["fps"]

        # Calcular el intervalo de frames para lograr sample_rate frames/segundo
        frame_interval = max(1, int(fps / sample_rate)) if fps > 0 else 1

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frames_saved = 0
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_interval == 0:
                    filename = output_path / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    frames_saved += 1

                frame_index += 1
        finally:
            cap.release()

        return frames_saved

    def resize_frame(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
    ) -> np.ndarray:
        """
        Redimensiona un frame al tamaño objetivo usando interpolación lineal.

        Args:
            frame: Array NumPy (H, W, C) en formato BGR.
            target_size: Tupla (ancho, alto) del tamaño de salida. Por defecto (640, 640).

        Returns:
            Array NumPy redimensionado al tamaño objetivo.
        """
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    preprocessor = VideoPreprocessor()
    print("VideoPreprocessor inicializado correctamente.")
