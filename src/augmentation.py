"""
augmentation.py
---------------
Generación de imágenes sintéticas mediante data augmentation.

Responsabilidades:
- Aplicar transformaciones geométricas y fotométricas a imágenes anotadas.
- Adaptar las anotaciones YOLO (bounding boxes) tras cada transformación.
- Generar un dataset sintético ampliado a partir del dataset base.
- Guardar imágenes y etiquetas aumentadas en data/synthetic/.
"""

import random
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np


# Pipeline de augmentation base para entrenamiento
DEFAULT_TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.RandomShadow(p=0.2),
        A.Rotate(limit=15, p=0.4),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
)


class SyntheticAugmentor:
    """Generador de imágenes sintéticas con augmentation para entrenamiento YOLO."""

    def __init__(
        self,
        transform: Optional[A.Compose] = None,
        output_dir: str = "data/synthetic",
        seed: int = 42,
    ):
        """
        Inicializa el augmentor.

        Args:
            transform: Pipeline de albumentations. Si es None, usa DEFAULT_TRANSFORM.
            output_dir: Directorio donde se guardan imagen y etiqueta aumentadas.
            seed: Semilla aleatoria para reproducibilidad.
        """
        self.transform = transform or DEFAULT_TRANSFORM
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)

    def load_yolo_labels(self, label_path: str) -> tuple[list[int], list[list[float]]]:
        """
        Carga un archivo de etiquetas en formato YOLO.

        Args:
            label_path: Ruta al archivo .txt de etiquetas.

        Returns:
            Tupla (class_ids, bboxes) donde bboxes es [[cx, cy, w, h], ...] en [0, 1].
        """
        pass

    def save_yolo_labels(
        self,
        label_path: str,
        class_ids: list[int],
        bboxes: list[list[float]],
    ) -> None:
        """
        Guarda etiquetas en formato YOLO.

        Args:
            label_path: Ruta de salida del archivo .txt.
            class_ids: Lista de IDs de clase.
            bboxes: Lista de [cx, cy, w, h] normalizados.
        """
        pass

    def augment_one(
        self,
        image: np.ndarray,
        class_ids: list[int],
        bboxes: list[list[float]],
    ) -> tuple[np.ndarray, list[int], list[list[float]]]:
        """
        Aplica el pipeline de augmentation a una imagen y sus bboxes.

        Args:
            image: Array BGR de la imagen.
            class_ids: IDs de clase de cada bbox.
            bboxes: Bounding boxes en formato YOLO normalizadas.

        Returns:
            Tupla (imagen_aumentada, class_ids_filtrados, bboxes_filtradas).
        """
        pass

    def generate_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        n_augmentations: int = 5,
    ) -> int:
        """
        Genera un dataset sintético aumentado a partir de un directorio de imágenes anotadas.

        Args:
            images_dir: Directorio con imágenes originales.
            labels_dir: Directorio con archivos .txt de etiquetas YOLO.
            n_augmentations: Número de versiones aumentadas por imagen.

        Returns:
            Número total de imágenes generadas.
        """
        pass


if __name__ == "__main__":
    augmentor = SyntheticAugmentor(output_dir="data/synthetic", seed=42)
    print("SyntheticAugmentor inicializado.")
    print(f"  Output dir: {augmentor.output_dir}")
    print(f"  Seed: {augmentor.seed}")
