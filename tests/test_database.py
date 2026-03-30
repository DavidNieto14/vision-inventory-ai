"""
test_database.py
----------------
Pruebas unitarias para el módulo src/database.py.
Usa directorios temporales de pytest para no afectar la base de datos real.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import InventoryDB

SAMPLE_DETECTIONS = [
    {"category": "CONFORME", "confidence": 0.92, "frame_number": 0},
    {"category": "VEC",      "confidence": 0.85, "frame_number": 0},
    {"category": "CONFORME", "confidence": 0.78, "frame_number": 1},
    {"category": "SCRAP",    "confidence": 0.91, "frame_number": 1},
]


@pytest.fixture
def db(tmp_path):
    """Crea una instancia de InventoryDB en un directorio temporal."""
    db_file = tmp_path / "exports" / "inventory.db"
    return InventoryDB(db_path=str(db_file))


class TestInventoryDB:
    """Suite de pruebas para InventoryDB."""

    def test_db_file_created_on_init(self, tmp_path):
        """El archivo .db se crea al inicializar InventoryDB."""
        db_file = tmp_path / "exports" / "test.db"
        InventoryDB(db_path=str(db_file))
        assert db_file.exists()

    def test_tables_created_on_init(self, db):
        """Las tablas 'detections' e 'inventory_summary' existen tras la inicialización."""
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "detections" in tables
        assert "inventory_summary" in tables

    def test_insert_detection_persists(self, db):
        """insert_detection() guarda un registro en la tabla detections."""
        db.insert_detection(
            batch_id="LOTE_TEST",
            category="CONFORME",
            confidence=0.92,
            frame_number=5,
        )
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.execute("SELECT * FROM detections WHERE batch_id='LOTE_TEST'")
        rows = cursor.fetchall()
        conn.close()
        assert len(rows) == 1

    def test_insert_detection_correct_values(self, db):
        """Los valores insertados coinciden con los argumentos pasados."""
        db.insert_detection(
            batch_id="LOTE_001",
            category="SCRAP",
            confidence=0.75,
            frame_number=10,
        )
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM detections WHERE batch_id='LOTE_001'")
        row = cursor.fetchone()
        conn.close()
        assert row["batch_id"] == "LOTE_001"
        assert row["category"] == "SCRAP"
        assert abs(row["confidence"] - 0.75) < 1e-6
        assert row["frame_number"] == 10

    def test_insert_multiple_detections(self, db):
        """Se pueden insertar múltiples detecciones para el mismo lote."""
        for det in SAMPLE_DETECTIONS:
            db.insert_detection(
                batch_id="LOTE_MULTI",
                category=det["category"],
                confidence=det["confidence"],
                frame_number=det["frame_number"],
            )
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM detections WHERE batch_id='LOTE_MULTI'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == len(SAMPLE_DETECTIONS)

    def test_update_summary_inserts_row(self, db):
        """update_summary() inserta un registro en inventory_summary."""
        counts = {"CONFORME": 10, "VEC": 2, "SCRAP": 1, "RETRABAJO": 0}
        db.update_summary(batch_id="LOTE_SUM", counts_dict=counts)
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.execute("SELECT * FROM inventory_summary WHERE batch_id='LOTE_SUM'")
        row = cursor.fetchone()
        conn.close()
        assert row is not None

    def test_update_summary_correct_counts(self, db):
        """Los conteos guardados en update_summary coinciden con los del diccionario."""
        counts = {"CONFORME": 5, "VEC": 3, "SCRAP": 2, "RETRABAJO": 1}
        db.update_summary(batch_id="LOTE_COUNTS", counts_dict=counts)
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM inventory_summary WHERE batch_id='LOTE_COUNTS'")
        row = cursor.fetchone()
        conn.close()
        assert row["conformes"] == 5
        assert row["vec"] == 3
        assert row["scrap"] == 2
        assert row["retrabajo"] == 1
        assert row["total"] == 11

    def test_update_summary_upserts(self, db):
        """update_summary() actualiza los conteos si el batch_id ya existe."""
        db.update_summary("LOTE_UPSERT", {"CONFORME": 1, "VEC": 0, "SCRAP": 0, "RETRABAJO": 0})
        db.update_summary("LOTE_UPSERT", {"CONFORME": 10, "VEC": 5, "SCRAP": 0, "RETRABAJO": 0})
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM inventory_summary WHERE batch_id='LOTE_UPSERT'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1  # Solo debe haber un registro tras el upsert

    def test_get_summary_returns_dataframe(self, db):
        """get_summary() retorna un objeto DataFrame."""
        db.update_summary("LOTE_DF", {"CONFORME": 1, "VEC": 0, "SCRAP": 0, "RETRABAJO": 0})
        result = db.get_summary("LOTE_DF")
        assert isinstance(result, pd.DataFrame)

    def test_get_summary_correct_columns(self, db):
        """El DataFrame de get_summary() contiene las columnas esperadas."""
        db.update_summary("LOTE_COLS", {"CONFORME": 1, "VEC": 0, "SCRAP": 0, "RETRABAJO": 0})
        df = db.get_summary("LOTE_COLS")
        expected = {"id", "timestamp", "batch_id", "conformes", "vec", "scrap", "retrabajo", "total"}
        assert expected.issubset(set(df.columns))

    def test_get_summary_empty_for_unknown_batch(self, db):
        """get_summary() retorna DataFrame vacío para un batch_id inexistente."""
        df = db.get_summary("LOTE_INEXISTENTE")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_export_csv_creates_file(self, db, tmp_path):
        """export_csv() crea el archivo CSV en la ruta indicada."""
        db.insert_detection("LOTE_CSV", "CONFORME", 0.9, 0)
        csv_path = tmp_path / "output.csv"
        db.export_csv(batch_id="LOTE_CSV", output_path=str(csv_path))
        assert csv_path.exists()

    def test_export_csv_correct_rows(self, db, tmp_path):
        """El CSV exportado contiene el mismo número de filas que las detecciones insertadas."""
        for det in SAMPLE_DETECTIONS:
            db.insert_detection("LOTE_CSV2", det["category"], det["confidence"], det["frame_number"])

        csv_path = tmp_path / "lote_csv2.csv"
        db.export_csv(batch_id="LOTE_CSV2", output_path=str(csv_path))
        df = pd.read_csv(str(csv_path))
        assert len(df) == len(SAMPLE_DETECTIONS)

    def test_export_csv_empty_for_unknown_batch(self, db, tmp_path):
        """export_csv() crea un CSV vacío (solo headers) para un lote sin detecciones."""
        csv_path = tmp_path / "empty.csv"
        db.export_csv(batch_id="LOTE_VACIO", output_path=str(csv_path))
        assert csv_path.exists()
        df = pd.read_csv(str(csv_path))
        assert len(df) == 0
