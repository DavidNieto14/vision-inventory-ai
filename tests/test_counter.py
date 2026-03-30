"""
test_counter.py
---------------
Pruebas unitarias para el módulo src/counter.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counter import PieceCounter, CATEGORIES, CLASS_ID_TO_CATEGORY

SAMPLE_DETECTIONS = [
    {"class_id": 0, "confidence": 0.92, "bbox": [10, 10, 50, 50]},
    {"class_id": 1, "confidence": 0.85, "bbox": [60, 60, 100, 100]},
    {"class_id": 0, "confidence": 0.78, "bbox": [120, 30, 160, 70]},
    {"class_id": 2, "confidence": 0.91, "bbox": [200, 200, 250, 250]},
    {"class_id": 3, "confidence": 0.65, "bbox": [300, 100, 350, 150]},
]


class TestPieceCounter:
    """Suite de pruebas para PieceCounter."""

    def test_initialization_default_threshold(self):
        """El contador se inicializa con umbral por defecto 0.5."""
        counter = PieceCounter()
        assert counter.conf_threshold == 0.5

    def test_initialization_custom_threshold(self):
        """El contador acepta umbral de confianza personalizado."""
        counter = PieceCounter(conf_threshold=0.7)
        assert counter.conf_threshold == 0.7

    def test_initial_counts_all_zero(self):
        """Todos los conteos son cero al inicializar."""
        counter = PieceCounter()
        counts = counter.get_counts()
        assert all(v == 0 for v in counts.values())

    def test_initial_counts_has_all_categories(self):
        """get_counts() retorna las cuatro categorías desde el inicio."""
        counter = PieceCounter()
        counts = counter.get_counts()
        for cat in CATEGORIES:
            assert cat in counts

    def test_update_increments_correct_category(self):
        """update() incrementa únicamente la categoría correspondiente al class_id."""
        counter = PieceCounter()
        counter.update([{"class_id": 0, "confidence": 0.9, "bbox": [0, 0, 10, 10]}])
        counts = counter.get_counts()
        assert counts["CONFORME"] == 1
        assert counts["VEC"] == 0
        assert counts["SCRAP"] == 0
        assert counts["RETRABAJO"] == 0

    def test_update_multiple_detections(self):
        """update() cuenta correctamente múltiples detecciones en un frame."""
        counter = PieceCounter()
        counter.update(SAMPLE_DETECTIONS)
        counts = counter.get_counts()
        assert counts["CONFORME"] == 2
        assert counts["VEC"] == 1
        assert counts["SCRAP"] == 1
        assert counts["RETRABAJO"] == 1

    def test_update_accumulates_across_calls(self):
        """Múltiples llamadas a update() acumulan correctamente los conteos."""
        counter = PieceCounter()
        counter.update(SAMPLE_DETECTIONS)
        counter.update(SAMPLE_DETECTIONS)
        counts = counter.get_counts()
        assert counts["CONFORME"] == 4
        assert counts["VEC"] == 2

    def test_update_filters_low_confidence(self):
        """Las detecciones con confidence < conf_threshold no se cuentan."""
        counter = PieceCounter(conf_threshold=0.8)
        low_conf = [{"class_id": 0, "confidence": 0.5, "bbox": [0, 0, 10, 10]}]
        counter.update(low_conf)
        counts = counter.get_counts()
        assert counts["CONFORME"] == 0

    def test_update_accepts_exactly_threshold(self):
        """Una detección con confidence == conf_threshold NO se cuenta (< estricto)."""
        counter = PieceCounter(conf_threshold=0.8)
        exact_conf = [{"class_id": 0, "confidence": 0.8, "bbox": [0, 0, 10, 10]}]
        counter.update(exact_conf)
        # confidence < threshold es la condición de rechazo, 0.8 >= 0.8 → debe contar
        counts = counter.get_counts()
        assert counts["CONFORME"] == 1

    def test_update_ignores_unknown_class_id(self):
        """Detecciones con class_id no mapeado son ignoradas silenciosamente."""
        counter = PieceCounter()
        unknown = [{"class_id": 99, "confidence": 0.95, "bbox": [0, 0, 10, 10]}]
        counter.update(unknown)
        counts = counter.get_counts()
        assert sum(counts.values()) == 0

    def test_update_empty_list(self):
        """update() con lista vacía no modifica los conteos."""
        counter = PieceCounter()
        counter.update([])
        counts = counter.get_counts()
        assert all(v == 0 for v in counts.values())

    def test_reset_clears_all_counts(self):
        """reset() reinicia todos los contadores a cero."""
        counter = PieceCounter()
        counter.update(SAMPLE_DETECTIONS)
        counter.reset()
        counts = counter.get_counts()
        assert all(v == 0 for v in counts.values())

    def test_reset_allows_new_accumulation(self):
        """Tras reset(), los conteos se acumulan desde cero correctamente."""
        counter = PieceCounter()
        counter.update(SAMPLE_DETECTIONS)
        counter.reset()
        counter.update([{"class_id": 2, "confidence": 0.9, "bbox": [0, 0, 10, 10]}])
        counts = counter.get_counts()
        assert counts["SCRAP"] == 1
        assert counts["CONFORME"] == 0

    def test_get_counts_returns_dict(self):
        """get_counts() retorna un diccionario."""
        counter = PieceCounter()
        assert isinstance(counter.get_counts(), dict)

    def test_class_id_to_category_mapping(self):
        """El mapeo CLASS_ID_TO_CATEGORY cubre los cuatro IDs esperados."""
        assert CLASS_ID_TO_CATEGORY[0] == "CONFORME"
        assert CLASS_ID_TO_CATEGORY[1] == "VEC"
        assert CLASS_ID_TO_CATEGORY[2] == "SCRAP"
        assert CLASS_ID_TO_CATEGORY[3] == "RETRABAJO"
