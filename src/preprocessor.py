"""
preprocessor.py
---------------
Preprocesamiento de imágenes y video antes de la inferencia.

Responsabilidades:
- Redimensionar y normalizar imágenes al tamaño esperado por el modelo.
- Aplicar correcciones de brillo, contraste y balance de color.
- Extraer frames de un video a una tasa configurable.
- Guardar frames preprocesados en disco.
"""

from pathlib import Path
from typing import Generator, Optional, Union

import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocesador de imágenes estáticas para inferencia con YOLOv8."""

    def __init__(self, target_size: tuple[int, int] = (640, 640)):
        """
        Inicializa el preprocesador.

        Args:
            target_size: Tamaño (ancho, alto) al que se redimensionarán las imágenes.
        """
        self.target_size = target_size

    def load_image(self, path: str) -> np.ndarray:
        """
        Carga una imagen desde disco en formato BGR.

        Args:
            path: Ruta al archivo de imagen.

        Returns:
            Array NumPy (H, W, 3) en BGR.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si la imagen no puede leerse.
        """
        pass

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensiona la imagen al tamaño objetivo manteniendo relación de aspecto (letterbox).

        Args:
            image: Array BGR de entrada.

        Returns:
            Array BGR redimensionado con padding gris.
        """
        pass

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de píxel al rango [0, 1].

        Args:
            image: Array BGR uint8.

        Returns:
            Array float32 en [0, 1].
        """
        pass

    def enhance_contrast(
        self, image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)
    ) -> np.ndarray:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) al canal L (LAB).

        Args:
            image: Array BGR uint8.
            clip_limit: Límite de recorte para CLAHE.
            tile_grid: Tamaño de la cuadrícula de tiles.

        Returns:
            Imagen con contraste mejorado en BGR.
        """
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline completo: resize → enhance_contrast.

        Args:
            image: Array BGR original.

        Returns:
            Array BGR preprocesado listo para el detector.
        """
        pass


class VideoFrameExtractor:
    """Extractor de frames desde archivos de video o streams."""

    def __init__(self, source: Union[int, str], fps_target: Optional[float] = None):
        """
        Inicializa el extractor.

        Args:
            source: Índice de cámara o ruta/URL al video.
            fps_target: FPS a los que extraer frames. Si es None, usa los FPS nativos del video.
        """
        self.source = source
        self.fps_target = fps_target
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Abre la fuente de video."""
        pass

    def close(self) -> None:
        """Libera el recurso de video."""
        pass

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Generador que produce (frame_id, frame_BGR) respetando fps_target.

        Yields:
            Tuplas (frame_id, array BGR).
        """
        pass

    def save_frames(self, output_dir: str, prefix: str = "frame") -> int:
        """
        Extrae y guarda todos los frames en disco.

        Args:
            output_dir: Directorio de salida.
            prefix: Prefijo de nombre para los archivos.

        Returns:
            Número total de frames guardados.
        """
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    preprocessor = ImagePreprocessor(target_size=(640, 640))
    print("ImagePreprocessor inicializado.")
    print(f"  Target size: {preprocessor.target_size}")

    extractor = VideoFrameExtractor(source=0)
    print("VideoFrameExtractor inicializado.")
