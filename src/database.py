"""
database.py
-----------
Módulo de persistencia SQLite para el sistema de inventario.

Responsabilidades:
- Crear y migrar el esquema de la base de datos.
- Insertar registros de detecciones individuales.
- Insertar resúmenes de sesiones de conteo.
- Consultar históricos y exportar a DataFrame/CSV.
- Gestionar la conexión de forma segura (context manager).
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Schema SQL
CREATE_DETECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    frame_id    INTEGER,
    timestamp   TEXT    NOT NULL,
    class_name  TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    bbox_x1     REAL,
    bbox_y1     REAL,
    bbox_x2     REAL,
    bbox_y2     REAL
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL UNIQUE,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    source      TEXT,
    total_count INTEGER DEFAULT 0
);
"""


class InventoryDatabase:
    """Interfaz SQLite para persistencia del inventario de piezas."""

    def __init__(self, db_path: str = None):
        """
        Inicializa la conexión a la base de datos.

        Args:
            db_path: Ruta al archivo .db. Si es None, usa DB_PATH del .env.
        """
        self.db_path = db_path or os.getenv("DB_PATH", "data/inventory.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Crea las tablas si no existen."""
        pass

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager que abre y cierra la conexión de forma segura."""
        pass

    def create_session(self, session_id: str, source: Optional[str] = None) -> None:
        """
        Registra el inicio de una nueva sesión de detección.

        Args:
            session_id: Identificador único de la sesión.
            source: Fuente de video o imagen (ruta, URL, índice de cámara).
        """
        pass

    def close_session(self, session_id: str, total_count: int) -> None:
        """
        Marca una sesión como finalizada y actualiza el conteo total.

        Args:
            session_id: Identificador de la sesión.
            total_count: Número total de piezas contadas.
        """
        pass

    def insert_detection(
        self,
        session_id: str,
        class_name: str,
        confidence: float,
        frame_id: Optional[int] = None,
        bbox: Optional[list[float]] = None,
    ) -> None:
        """
        Inserta una detección individual.

        Args:
            session_id: Sesión a la que pertenece la detección.
            class_name: Nombre de la clase detectada.
            confidence: Confianza de la detección.
            frame_id: Número de frame (opcional).
            bbox: [x1, y1, x2, y2] en píxeles (opcional).
        """
        pass

    def insert_detections_batch(
        self, session_id: str, detections: list[dict], frame_id: Optional[int] = None
    ) -> None:
        """
        Inserta múltiples detecciones en una sola transacción.

        Args:
            session_id: Sesión a la que pertenecen.
            detections: Salida de PieceDetector.detect().
            frame_id: Número de frame (opcional).
        """
        pass

    def query_session(self, session_id: str) -> pd.DataFrame:
        """
        Retorna todas las detecciones de una sesión como DataFrame.

        Args:
            session_id: Identificador de la sesión.

        Returns:
            DataFrame con todas las columnas de la tabla detections.
        """
        pass

    def query_summary(self) -> pd.DataFrame:
        """
        Retorna un resumen agregado de todas las sesiones.

        Returns:
            DataFrame con columnas: session_id, class_name, total_count.
        """
        pass


if __name__ == "__main__":
    db = InventoryDatabase()
    print("InventoryDatabase inicializada correctamente.")
    print(f"  DB path: {db.db_path}")
