"""
test_database.py
----------------
Pruebas unitarias para el módulo src/database.py.
Usa una base de datos en memoria o en tmp_path para no afectar datos reales.
"""

import pytest


SAMPLE_DETECTIONS = [
    {"class_id": 0, "class_name": "pieza_a", "confidence": 0.92, "bbox": [10, 10, 50, 50]},
    {"class_id": 1, "class_name": "pieza_b", "confidence": 0.85, "bbox": [60, 60, 100, 100]},
]


class TestInventoryDatabase:
    """Suite de pruebas para InventoryDatabase."""

    @pytest.fixture
    def db(self, tmp_path):
        """Crea una instancia de BD en un directorio temporal."""
        pass

    def test_tables_created_on_init(self, db):
        """Las tablas 'detections' y 'sessions' existen tras la inicialización."""
        pass

    def test_create_session(self, db):
        """create_session() inserta un registro en la tabla sessions."""
        pass

    def test_create_duplicate_session_raises(self, db):
        """Crear dos sesiones con el mismo ID lanza un error de integridad."""
        pass

    def test_close_session_updates_ended_at(self, db):
        """close_session() actualiza el campo ended_at y total_count."""
        pass

    def test_insert_detection(self, db):
        """insert_detection() persiste una detección correctamente."""
        pass

    def test_insert_detections_batch(self, db):
        """insert_detections_batch() inserta todas las detecciones del lote."""
        pass

    def test_query_session_returns_dataframe(self, db):
        """query_session() retorna un DataFrame con las columnas correctas."""
        pass

    def test_query_session_empty(self, db):
        """query_session() retorna DataFrame vacío para sesión sin detecciones."""
        pass

    def test_query_summary_aggregates_correctly(self, db):
        """query_summary() retorna totales correctos por clase."""
        pass
