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

# Nombres de clases COCO relevantes para visualización
COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 39: "bottle", 41: "cup",
    56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 62: "tv", 63: "laptop",
    67: "cell phone", 72: "tv",
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

    def detect_frame(
        self,
        frame: np.ndarray,
        roi: Optional[tuple] = None,
    ) -> List[dict]:
        """
        Ejecuta detección sobre un frame individual, con soporte opcional de ROI.

        Args:
            frame: Array NumPy (H, W, 3) en formato BGR.
            roi: Tupla (x1, y1, x2, y2) en píxeles que define la región de interés.
                 Si se provee, la inferencia se ejecuta sobre el recorte del ROI y
                 las coordenadas se retraducen al espacio del frame completo.
                 Se descartan detecciones cuyo centro quede fuera del ROI.

        Returns:
            Lista de diccionarios, uno por detección, con claves:
                - 'class_id'      (int): ID numérico de la clase.
                - 'confidence'    (float): Confianza de la predicción.
                - 'bbox'          (list[float]): [x1, y1, x2, y2] en píxeles del frame completo.
                - 'category_name' (str): Nombre de la categoría según CLASS_NAMES / COCO_NAMES.
        """
        if self.model is None:
            raise RuntimeError("El modelo no está cargado. Llame a load_model() primero.")

        if roi is not None:
            rx1, ry1, rx2, ry2 = (int(v) for v in roi)
            h, w = frame.shape[:2]
            rx1, ry1 = max(rx1, 0), max(ry1, 0)
            rx2, ry2 = min(rx2, w), min(ry2, h)
            source = frame[ry1:ry2, rx1:rx2]
        else:
            source = frame

        results = self.model.predict(
            source=source,
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

                # Retransladar coordenadas al espacio del frame completo
                if roi is not None:
                    x1 += rx1; y1 += ry1
                    x2 += rx1; y2 += ry1
                    # Descartar si el centro queda fuera del ROI
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue

                category_name = (
                    CLASS_NAMES.get(class_id)
                    or COCO_NAMES.get(class_id)
                    or f"objeto_clase_{class_id}"
                )
                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": round(confidence, 4),
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "category_name": category_name,
                    }
                )

        return detections

    # Colores BGR por categoría
    _CATEGORY_COLORS: Dict[str, tuple] = {
        "CONFORME": (0, 255, 0),
        "VEC":      (0, 255, 255),
        "SCRAP":    (0, 0, 255),
        "RETRABAJO":(0, 165, 255),
    }

    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[dict],
        roi: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Dibuja bounding boxes y etiquetas sobre el frame para cada detección.

        Para cada detección dibuja:
        - Un rectángulo de color según la categoría.
        - Un recuadro semitransparente detrás de la etiqueta.
        - El nombre de la categoría y el confidence score.

        Args:
            frame: Array NumPy (H, W, 3) BGR original (no se modifica in-place).
            detections: Lista de dicts retornada por detect_frame().
            roi: Tupla opcional (x1, y1, x2, y2). Si se provee, dibuja el rectángulo
                 del ROI en azul sobre el frame anotado.

        Returns:
            Nuevo frame anotado como numpy array (H, W, 3) BGR.
        """
        annotated = frame.copy()
        overlay = frame.copy()

        # Dibujar ROI en azul antes de las bboxes
        if roi is not None:
            rx1, ry1, rx2, ry2 = (int(v) for v in roi)
            # Rectángulo punteado: líneas discontinuas dibujadas manualmente
            dash_len, gap_len = 12, 6
            roi_color = (255, 80, 0)  # azul BGR
            for side in ["top", "bottom", "left", "right"]:
                if side == "top":
                    pts = [(x, ry1) for x in range(rx1, rx2, dash_len + gap_len)]
                    for sx in pts:
                        cv2.line(annotated, sx, (min(sx[0] + dash_len, rx2), ry1), roi_color, 2)
                elif side == "bottom":
                    pts = [(x, ry2) for x in range(rx1, rx2, dash_len + gap_len)]
                    for sx in pts:
                        cv2.line(annotated, sx, (min(sx[0] + dash_len, rx2), ry2), roi_color, 2)
                elif side == "left":
                    pts = [(rx1, y) for y in range(ry1, ry2, dash_len + gap_len)]
                    for sy in pts:
                        cv2.line(annotated, sy, (rx1, min(sy[1] + dash_len, ry2)), roi_color, 2)
                elif side == "right":
                    pts = [(rx2, y) for y in range(ry1, ry2, dash_len + gap_len)]
                    for sy in pts:
                        cv2.line(annotated, sy, (rx2, min(sy[1] + dash_len, ry2)), roi_color, 2)
            cv2.putText(annotated, "ROI", (rx1 + 4, ry1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)
        overlay = annotated.copy()

        for det in detections:
            category = det["category_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            color = self._CATEGORY_COLORS.get(category, (200, 200, 200))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

            # Etiqueta con confidence
            label = f"{category} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Recuadro semitransparente detrás del texto
            pad = 3
            tx1, ty1 = x1, max(y1 - th - baseline - pad * 2, 0)
            tx2, ty2 = x1 + tw + pad * 2, max(y1, th + baseline + pad * 2)
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, thickness=-1)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
            # Actualizar overlay para siguiente iteración
            overlay = annotated.copy()

            cv2.putText(
                annotated,
                label,
                (x1 + pad, max(y1 - baseline - pad, th)),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    def process_video(
        self,
        video_path: str,
        batch_id: str,
        db,
        counter,
        roi: Optional[tuple] = None,
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
            roi: Tupla opcional (x1, y1, x2, y2) que define la región de interés para detecciones.
                 Si se provee, solo se consideran detecciones dentro de esta región.

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

                detections = self.detect_frame(frame, roi=roi)
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
