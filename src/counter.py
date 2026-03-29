"""
counter.py
----------
Lógica de conteo de piezas por categoría a partir de los resultados del detector.

Responsabilidades:
- Agregar detecciones por clase en un frame o secuencia de frames.
- Mantener un conteo acumulado durante una sesión de video.
- Exportar resúmenes de conteo a diccionario o DataFrame.
- Resetear contadores entre sesiones.
"""

from collections import defaultdict
from typing import Optional

import pandas as pd


class PieceCounter:
    """Contador acumulado de piezas detectadas por categoría."""

    def __init__(self, class_names: Optional[list[str]] = None):
        """
        Inicializa el contador.

        Args:
            class_names: Lista ordenada de nombres de clases del modelo.
                         Si es None se infieren de los resultados en tiempo de ejecución.
        """
        self.class_names = class_names or []
        self._counts: dict[str, int] = defaultdict(int)
        self._frame_history: list[dict] = []

    def count_from_results(self, detections: list[dict]) -> dict[str, int]:
        """
        Cuenta las piezas presentes en una lista de detecciones de un solo frame.

        Args:
            detections: Salida de PieceDetector.detect().

        Returns:
            Diccionario {class_name: count} para ese frame.
        """
        pass

    def update(self, detections: list[dict], frame_id: Optional[int] = None) -> None:
        """
        Actualiza los contadores acumulados con las detecciones del frame actual.

        Args:
            detections: Salida de PieceDetector.detect().
            frame_id: Identificador del frame (opcional, para trazabilidad).
        """
        pass

    def get_totals(self) -> dict[str, int]:
        """
        Retorna el conteo acumulado total de la sesión.

        Returns:
            Diccionario {class_name: total_count}.
        """
        pass

    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Retorna el historial de conteos por frame como DataFrame.

        Returns:
            DataFrame con columnas: frame_id, class_name, count.
        """
        pass

    def reset(self) -> None:
        """Reinicia todos los contadores y el historial."""
        pass

    def export_csv(self, output_path: str) -> None:
        """
        Exporta el historial de conteos a un archivo CSV.

        Args:
            output_path: Ruta del archivo CSV de salida.
        """
        pass


if __name__ == "__main__":
    counter = PieceCounter(class_names=["pieza_a", "pieza_b", "pieza_c"])
    print("PieceCounter inicializado correctamente.")
    print(f"  Clases registradas: {counter.class_names}")
