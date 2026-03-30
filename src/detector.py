"""
detector.py
-----------
Módulo principal de detección de piezas industriales usando YOLOv8.

Responsabilidades:
- Cargar el modelo YOLOv8 desde disco o descargarlo si no existe.
- Ejecutar inferencia sobre frames individuales (numpy BGR).
- Procesar videos completos frame a frame integrando counter y database.
- Retornar resultados estructurados con class_id, confidence, bbox y category_name.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Mapeo de class_id a nombre de categoría del dominio
CLASS_NAMES: Dict[int, str] = {
    0: "CONFORME",
    1: "VEC",
    2: "SCRAP",
    3: "RETRABAJO",
}


class PieceDetector:
    """
    Detector de piezas industriales basado en YOLOv8.

    Encapsula la carga del modelo, la inferencia sobre frames y el
    procesamiento completo de videos con integración a base de datos y contador.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Inicializa el detector con los parámetros de configuración.

        Args:
            model_path: Ruta al archivo de pesos YOLOv8 (.pt).
                        Si no existe, se descarga yolov8n.pt automáticamente.
            conf_threshold: Umbral de confianza mínimo para aceptar detecciones.
            device: Dispositivo de inferencia ('cpu', 'cuda', 'mps').
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """
        Carga el modelo YOLOv8 desde disco usando la librería ultralytics.

        Si el archivo de pesos no existe en model_path, descarga yolov8n.pt
        desde los servidores de Ultralytics y lo guarda en el directorio models/.

        Raises:
            RuntimeError: Si el modelo no puede cargarse ni descargarse.
        """
        model_file = Path(self.model_path)

        if not model_file.exists():
            print(f"[INFO] Modelo no encontrado en '{self.model_path}'. Descargando yolov8n.pt...")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            # Ultralytics descarga automáticamente si se pasa solo el nombre
            self.model = YOLO("yolov8n.pt")
            # Guardar los pesos descargados en la ruta configurada
            self.model.save(str(model_file))
        else:
            self.model = YOLO(str(model_file))

        print(f"[INFO] Modelo cargado: {self.model_path} | device={self.device}")

    def detect_frame(self, frame: np.ndarray) -> List[dict]:
        """
        Ejecuta detección sobre un frame individual.

        Args:
            frame: Array NumPy (H, W, 3) en formato BGR.

        Returns:
            Lista de diccionarios, uno por detección, con claves:
                - 'class_id'     (int): ID numérico de la clase.
                - 'confidence'   (float): Confianza de la predicción.
                - 'bbox'         (list[float]): [x1, y1, x2, y2] en píxeles.
                - 'category_name' (str): Nombre de la categoría según CLASS_NAMES.
        """
        if self.model is None:
            raise RuntimeError("El modelo no está cargado. Llame a load_model() primero.")

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                category_name = CLASS_NAMES.get(class_id, f"CLASE_{class_id}")
                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": round(confidence, 4),
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "category_name": category_name,
                    }
                )

        return detections

    def process_video(
        self,
        video_path: str,
        batch_id: str,
        db,
        counter,
    ) -> dict:
        """
        Procesa un video completo frame a frame integrando detección, conteo y persistencia.

        Para cada frame:
          1. Ejecuta detect_frame() para obtener las detecciones.
          2. Actualiza el contador acumulado con counter.update().
          3. Persiste cada detección en la base de datos con db.insert_detection().
        Al finalizar actualiza el resumen en db con db.update_summary().

        Args:
            video_path: Ruta al archivo de video (.mp4 u otro formato compatible con OpenCV).
            batch_id: Identificador del lote de producción para este video.
            db: Instancia de InventoryDB para persistencia.
            counter: Instancia de PieceCounter para conteo acumulado.

        Returns:
            Diccionario con métricas finales:
                - 'batch_id'        (str): ID del lote procesado.
                - 'total_frames'    (int): Número total de frames procesados.
                - 'total_detections'(int): Total de detecciones en el video.
                - 'counts'          (dict): Conteo final por categoría.
                - 'duration_seconds'(float): Tiempo de procesamiento en segundos.

        Raises:
            FileNotFoundError: Si el archivo de video no existe.
            RuntimeError: Si el video no puede abrirse con OpenCV.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        counter.reset()
        total_frames = 0
        total_detections = 0
        start_time = time.time()

        try:
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detect_frame(frame)
                counter.update(detections)

                for det in detections:
                    db.insert_detection(
                        batch_id=batch_id,
                        category=det["category_name"],
                        confidence=det["confidence"],
                        frame_number=frame_number,
                    )

                total_detections += len(detections)
                frame_number += 1
                total_frames += 1
        finally:
            cap.release()

        counts = counter.get_counts()
        db.update_summary(batch_id=batch_id, counts_dict=counts)

        return {
            "batch_id": batch_id,
            "total_frames": total_frames,
            "total_detections": total_detections,
            "counts": counts,
            "duration_seconds": round(time.time() - start_time, 2),
        }


if __name__ == "__main__":
    detector = PieceDetector()
    print("PieceDetector inicializado correctamente.")
    print(f"  Modelo          : {detector.model_path}")
    print(f"  Conf. threshold : {detector.conf_threshold}")
    print(f"  Device          : {detector.device}")
    print(f"  Clases          : {CLASS_NAMES}")
