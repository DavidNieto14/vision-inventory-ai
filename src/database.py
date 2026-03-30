"""
database.py
-----------
Módulo de persistencia SQLite para el sistema de inventario de visión por computadora.

Responsabilidades:
- Crear y gestionar el esquema de la base de datos.
- Insertar registros de detecciones individuales por frame.
- Actualizar el resumen de inventario por lote (batch).
- Consultar resúmenes y exportar a DataFrame/CSV.
- Gestionar la conexión de forma segura mediante context manager.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

# Ruta por defecto de la base de datos
DEFAULT_DB_PATH = "data/exports/inventory.db"


class InventoryDB:
    """Interfaz SQLite para persistencia del inventario de piezas industriales."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Inicializa la conexión a la base de datos y crea las tablas si no existen.

        Args:
            db_path: Ruta al archivo .db. Por defecto usa data/exports/inventory.db.
        """
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.create_tables()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager que abre y cierra la conexión SQLite de forma segura.

        Yields:
            Conexión activa a la base de datos.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_tables(self) -> None:
        """
        Crea las tablas de la base de datos si no existen.

        Tablas:
            - detections: registro de cada detección individual por frame.
            - inventory_summary: resumen agregado de conteos por lote.
        """
        create_detections = """
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            batch_id     TEXT    NOT NULL,
            category     TEXT    NOT NULL,
            confidence   REAL    NOT NULL,
            frame_number INTEGER NOT NULL
        );
        """
        create_summary = """
        CREATE TABLE IF NOT EXISTS inventory_summary (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            batch_id   TEXT    NOT NULL UNIQUE,
            conformes  INTEGER DEFAULT 0,
            vec        INTEGER DEFAULT 0,
            scrap      INTEGER DEFAULT 0,
            retrabajo  INTEGER DEFAULT 0,
            total      INTEGER DEFAULT 0
        );
        """
        with self._get_connection() as conn:
            conn.execute(create_detections)
            conn.execute(create_summary)

    def insert_detection(
        self,
        batch_id: str,
        category: str,
        confidence: float,
        frame_number: int,
    ) -> None:
        """
        Inserta un registro de detección individual en la base de datos.

        Args:
            batch_id: Identificador del lote de producción.
            category: Nombre de la categoría detectada (CONFORME, VEC, SCRAP, RETRABAJO).
            confidence: Nivel de confianza de la detección (0.0 – 1.0).
            frame_number: Número de frame de donde proviene la detección.
        """
        sql = """
        INSERT INTO detections (timestamp, batch_id, category, confidence, frame_number)
        VALUES (?, ?, ?, ?, ?)
        """
        timestamp = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(sql, (timestamp, batch_id, category, confidence, frame_number))

    def update_summary(self, batch_id: str, counts_dict: dict) -> None:
        """
        Inserta o actualiza el resumen de inventario para un lote dado.

        Si el lote ya existe en la tabla, sus conteos se reemplazan por los nuevos valores.

        Args:
            batch_id: Identificador del lote de producción.
            counts_dict: Diccionario con conteos por categoría. Claves esperadas:
                         'CONFORME', 'VEC', 'SCRAP', 'RETRABAJO'.
        """
        conformes = counts_dict.get("CONFORME", 0)
        vec = counts_dict.get("VEC", 0)
        scrap = counts_dict.get("SCRAP", 0)
        retrabajo = counts_dict.get("RETRABAJO", 0)
        total = conformes + vec + scrap + retrabajo
        timestamp = datetime.now().isoformat()

        sql = """
        INSERT INTO inventory_summary (timestamp, batch_id, conformes, vec, scrap, retrabajo, total)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(batch_id) DO UPDATE SET
            timestamp = excluded.timestamp,
            conformes = excluded.conformes,
            vec       = excluded.vec,
            scrap     = excluded.scrap,
            retrabajo = excluded.retrabajo,
            total     = excluded.total
        """
        with self._get_connection() as conn:
            conn.execute(sql, (timestamp, batch_id, conformes, vec, scrap, retrabajo, total))

    def get_summary(self, batch_id: Optional[str] = None) -> pd.DataFrame:
        """
        Retorna el resumen de inventario como DataFrame.

        Args:
            batch_id: Si se proporciona, filtra por ese lote. Si es None, retorna todos.

        Returns:
            DataFrame con columnas: id, timestamp, batch_id, conformes, vec, scrap, retrabajo, total.
        """
        if batch_id:
            sql = "SELECT * FROM inventory_summary WHERE batch_id = ?"
            params = (batch_id,)
        else:
            sql = "SELECT * FROM inventory_summary ORDER BY timestamp DESC"
            params = ()

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        return pd.DataFrame([dict(row) for row in rows], columns=columns) if rows else pd.DataFrame(columns=columns)

    def export_csv(self, batch_id: str, output_path: str) -> None:
        """
        Exporta las detecciones de un lote a un archivo CSV.

        Args:
            batch_id: Identificador del lote a exportar.
            output_path: Ruta completa del archivo CSV de salida.
        """
        sql = "SELECT * FROM detections WHERE batch_id = ? ORDER BY frame_number"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (batch_id,))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        df = pd.DataFrame([dict(row) for row in rows], columns=columns) if rows else pd.DataFrame(columns=columns)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    db = InventoryDB()
    print("InventoryDB inicializada correctamente.")
    print(f"  DB path: {db.db_path}")
