"""
counter.py
----------
Lógica de conteo de piezas por categoría en una línea de pintura industrial.

Responsabilidades:
- Mantener conteos acumulados por categoría durante una sesión de video.
- Filtrar detecciones por umbral de confianza mínimo configurable.
- Mapear class_id numérico a nombre de categoría.
- Reiniciar contadores entre lotes o sesiones.
"""

from collections import defaultdict
from typing import Dict, List, Optional

# Mapeo de class_id a nombre de categoría
CLASS_ID_TO_CATEGORY: Dict[int, str] = {
    0: "CONFORME",
    1: "VEC",
    2: "SCRAP",
    3: "RETRABAJO",
}

# Categorías válidas del sistema
CATEGORIES = ("CONFORME", "VEC", "SCRAP", "RETRABAJO")


class PieceCounter:
    """
    Contador acumulado de piezas detectadas por categoría.

    Filtra detecciones cuya confianza sea menor al umbral configurado
    y acumula conteos por sesión. Soporta reset entre lotes.
    """

    def __init__(self, conf_threshold: float = 0.5):
        """
        Inicializa el contador con todas las categorías en cero.

        Args:
            conf_threshold: Umbral de confianza mínimo para aceptar una detección.
                            Detecciones con confidence < conf_threshold son ignoradas.
                            Por defecto 0.5.
        """
        self.conf_threshold = conf_threshold
        self._counts: Dict[str, int] = defaultdict(int, {cat: 0 for cat in CATEGORIES})

    def update(self, detections_list: List[dict]) -> None:
        """
        Actualiza los contadores acumulados con las detecciones de un frame.

        Solo se contabilizan las detecciones cuya confianza supere el umbral.
        Las detecciones con class_id desconocido son ignoradas.

        Args:
            detections_list: Lista de diccionarios con claves:
                - 'class_id'   (int): ID de clase del modelo.
                - 'confidence' (float): Confianza de la predicción.
                - 'bbox'       (list): Coordenadas [x1, y1, x2, y2].
        """
        for det in detections_list:
            confidence = det.get("confidence", 0.0)
            if confidence < self.conf_threshold:
                continue
            class_id = det.get("class_id")
            category = CLASS_ID_TO_CATEGORY.get(class_id)
            if category is None:
                continue
            self._counts[category] += 1

    def get_counts(self) -> Dict[str, int]:
        """
        Retorna el conteo acumulado actual por categoría.

        Returns:
            Diccionario con claves CONFORME, VEC, SCRAP, RETRABAJO
            y sus respectivos conteos acumulados.
        """
        return dict(self._counts)

    def reset(self) -> None:
        """
        Reinicia todos los contadores a cero.

        Debe llamarse al inicio de cada nuevo lote o sesión de procesamiento.
        """
        self._counts = defaultdict(int, {cat: 0 for cat in CATEGORIES})


if __name__ == "__main__":
    counter = PieceCounter()
    print("PieceCounter inicializado correctamente.")
    print(f"  Umbral de confianza: {counter.conf_threshold}")
    print(f"  Categorías: {list(CATEGORIES)}")
    print(f"  Conteos iniciales: {counter.get_counts()}")
