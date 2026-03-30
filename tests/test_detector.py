"""
test_detector.py
----------------
Pruebas unitarias para el módulo src/detector.py.

Las pruebas que requieren el modelo YOLO usan mocks para evitar
descargas de red y dependencias de hardware en CI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import PieceDetector, CLASS_NAMES


def make_mock_box(class_id: int, confidence: float, bbox: list):
    """Crea un mock de caja de detección YOLO con los valores indicados."""
    box = MagicMock()
    box.cls.item.return_value = class_id
    box.conf.item.return_value = confidence
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = bbox
    return box


def make_mock_result(detections: list):
    """Crea un mock de resultado YOLO con las detecciones dadas."""
    result = MagicMock()
    result.boxes = [
        make_mock_box(d["class_id"], d["confidence"], d["bbox"])
        for d in detections
    ]
    return result


@pytest.fixture
def mock_yolo_model():
    """Mock del modelo YOLO que retorna resultados vacíos por defecto."""
    mock = MagicMock()
    mock.predict.return_value = [make_mock_result([])]
    return mock


@pytest.fixture
def detector(tmp_path, mock_yolo_model):
    """PieceDetector con modelo mockeado para pruebas sin YOLO real."""
    model_file = tmp_path / "yolov8n.pt"
    model_file.touch()

    with patch("detector.YOLO", return_value=mock_yolo_model):
        d = PieceDetector(
            model_path=str(model_file),
            conf_threshold=0.5,
            device="cpu",
        )
    d.model = mock_yolo_model
    return d


class TestClassNames:
    """Pruebas sobre la constante CLASS_NAMES."""

    def test_class_names_has_four_entries(self):
        """CLASS_NAMES contiene exactamente cuatro categorías."""
        assert len(CLASS_NAMES) == 4

    def test_class_names_correct_mapping(self):
        """El mapeo de IDs a nombres es correcto."""
        assert CLASS_NAMES[0] == "CONFORME"
        assert CLASS_NAMES[1] == "VEC"
        assert CLASS_NAMES[2] == "SCRAP"
        assert CLASS_NAMES[3] == "RETRABAJO"


class TestPieceDetectorInit:
    """Pruebas de inicialización de PieceDetector."""

    def test_initialization_stores_params(self, tmp_path):
        """Los parámetros del constructor se almacenan correctamente."""
        model_file = tmp_path / "model.pt"
        model_file.touch()
        with patch("detector.YOLO"):
            d = PieceDetector(
                model_path=str(model_file),
                conf_threshold=0.7,
                device="cpu",
            )
        assert d.conf_threshold == 0.7
        assert d.device == "cpu"

    def test_model_not_found_downloads(self, tmp_path):
        """Si el modelo no existe, se intenta descargar yolov8n.pt."""
        non_existent = str(tmp_path / "noexiste.pt")
        mock_yolo = MagicMock()
        with patch("detector.YOLO", return_value=mock_yolo) as mock_cls:
            PieceDetector(model_path=non_existent, conf_threshold=0.5, device="cpu")
        # YOLO debe haberse llamado con 'yolov8n.pt' para la descarga
        mock_cls.assert_called_once_with("yolov8n.pt")


class TestDetectFrame:
    """Pruebas del método detect_frame()."""

    def test_detect_returns_list(self, detector, mock_yolo_model):
        """detect_frame() retorna una lista."""
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect_frame(frame)
        assert isinstance(result, list)

    def test_detect_empty_when_no_detections(self, detector, mock_yolo_model):
        """detect_frame() retorna lista vacía si no hay detecciones."""
        mock_yolo_model.predict.return_value = [make_mock_result([])]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect_frame(frame)
        assert result == []

    def test_detect_result_has_required_keys(self, detector, mock_yolo_model):
        """Cada detección contiene las claves: class_id, confidence, bbox, category_name."""
        mock_yolo_model.predict.return_value = [
            make_mock_result([{"class_id": 0, "confidence": 0.9, "bbox": [10, 10, 50, 50]}])
        ]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect_frame(frame)
        assert len(results) == 1
        det = results[0]
        assert "class_id" in det
        assert "confidence" in det
        assert "bbox" in det
        assert "category_name" in det

    def test_detect_confidence_in_range(self, detector, mock_yolo_model):
        """La confianza de cada detección está en el rango [0, 1]."""
        mock_yolo_model.predict.return_value = [
            make_mock_result([
                {"class_id": 0, "confidence": 0.85, "bbox": [0, 0, 10, 10]},
                {"class_id": 1, "confidence": 0.55, "bbox": [20, 20, 40, 40]},
            ])
        ]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect_frame(frame)
        for det in results:
            assert 0.0 <= det["confidence"] <= 1.0

    def test_detect_bbox_has_four_values(self, detector, mock_yolo_model):
        """Cada bbox contiene exactamente 4 valores [x1, y1, x2, y2]."""
        mock_yolo_model.predict.return_value = [
            make_mock_result([{"class_id": 2, "confidence": 0.9, "bbox": [5, 10, 60, 80]}])
        ]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect_frame(frame)
        assert len(results[0]["bbox"]) == 4

    def test_detect_category_name_from_class_names(self, detector, mock_yolo_model):
        """category_name coincide con el valor de CLASS_NAMES para el class_id dado."""
        mock_yolo_model.predict.return_value = [
            make_mock_result([{"class_id": 3, "confidence": 0.9, "bbox": [0, 0, 10, 10]}])
        ]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect_frame(frame)
        assert results[0]["category_name"] == CLASS_NAMES[3]

    def test_detect_multiple_detections(self, detector, mock_yolo_model):
        """detect_frame() retorna tantas detecciones como boxes en el resultado."""
        mock_dets = [
            {"class_id": 0, "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class_id": 1, "confidence": 0.8, "bbox": [20, 20, 30, 30]},
            {"class_id": 2, "confidence": 0.7, "bbox": [40, 40, 50, 50]},
        ]
        mock_yolo_model.predict.return_value = [make_mock_result(mock_dets)]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect_frame(frame)
        assert len(results) == 3

    def test_detect_unknown_class_id_uses_fallback_name(self, detector, mock_yolo_model):
        """class_id no mapeado recibe nombre de fallback 'CLASE_<id>'."""
        mock_yolo_model.predict.return_value = [
            make_mock_result([{"class_id": 99, "confidence": 0.9, "bbox": [0, 0, 10, 10]}])
        ]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect_frame(frame)
        assert results[0]["category_name"] == "CLASE_99"

    def test_detect_raises_without_model(self):
        """detect_frame() lanza RuntimeError si el modelo no está cargado."""
        d = object.__new__(PieceDetector)
        d.model = None
        d.conf_threshold = 0.5
        d.device = "cpu"
        d.model_path = "fake.pt"
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            d.detect_frame(frame)
