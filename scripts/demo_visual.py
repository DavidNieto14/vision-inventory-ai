"""
demo_visual.py
--------------
Demo de visualización con ROI calibrado para resolución 3040x1368.

Procesa video_linea_01.mp4 con conf=0.30 y el ROI definido en
model_config.yaml, guarda los primeros 5 frames con detecciones
dentro del ROI como figura3_deteccion_0X.jpg en data/exports/.

Uso:
    python scripts/demo_visual.py
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import PieceDetector

RAW_DIR        = ROOT / "data" / "raw"
EXPORTS_DIR    = ROOT / "data" / "exports"
CONF_THRESHOLD = 0.30
MAX_SAVED      = 5

# Colores BGR por categoría (grosor aumentado)
CATEGORY_COLORS = {
    "CONFORME":  (0, 255, 0),
    "VEC":       (0, 255, 255),
    "SCRAP":     (0, 0, 255),
    "RETRABAJO": (0, 165, 255),
}


def cargar_model_cfg() -> dict:
    cfg_path = ROOT / "configs" / "model_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model_path"] = str(ROOT / cfg["model_path"])
    return cfg


def draw_frame(
    frame: np.ndarray,
    detections: List[dict],
    roi: Optional[tuple] = None,
) -> np.ndarray:
    """
    Dibuja el ROI en azul grueso y las bounding boxes con texto grande.
    Texto escala 1.2, thickness=2. Bboxes thickness=3.
    """
    annotated = frame.copy()
    overlay   = frame.copy()

    # ── ROI: rectángulo azul sólido grueso ───────────────────────
    if roi is not None:
        rx1, ry1, rx2, ry2 = (int(v) for v in roi)
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 80, 0), thickness=3)
        cv2.putText(
            annotated, "ROI",
            (rx1 + 8, ry1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 0), 2, cv2.LINE_AA,
        )

    # ── Bounding boxes con texto grande ──────────────────────────
    for det in detections:
        category = det["category_name"]
        conf     = det["confidence"]
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        color = CATEGORY_COLORS.get(category, (200, 200, 200))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=3)

        label = f"{category} {conf:.2f}"
        font_scale = 1.2
        thickness  = 2
        font       = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        pad  = 5
        tx1  = x1
        ty1  = max(y1 - th - baseline - pad * 2, 0)
        tx2  = x1 + tw + pad * 2
        ty2  = max(y1, th + baseline + pad * 2)
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, thickness=-1)
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
        overlay = annotated.copy()

        cv2.putText(
            annotated, label,
            (x1 + pad, max(y1 - baseline - pad, th + pad)),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return annotated


def main() -> None:
    print("\n" + "=" * 60)
    print("  DEMO VISUAL con ROI — vision-inventory-ai")
    print("=" * 60)

    cfg = cargar_model_cfg()

    roi_cfg = cfg.get("roi", {})
    roi = None
    if roi_cfg.get("enabled", False):
        roi = (roi_cfg["x1"], roi_cfg["y1"], roi_cfg["x2"], roi_cfg["y2"])
        print(f"\n[ROI]   enabled=true  ({roi[0]},{roi[1]}) → ({roi[2]},{roi[3]})")
        print(f"        zona: {roi[2]-roi[0]}×{roi[3]-roi[1]} px sobre 3040×1368")

    print(f"[INIT]  modelo : {cfg['model_path']}")
    print(f"[INIT]  conf   : {CONF_THRESHOLD}  (override demo)")
    print(f"[INIT]  device : {cfg['device']}")

    detector = PieceDetector(
        model_path=cfg["model_path"],
        conf_threshold=CONF_THRESHOLD,
        device=cfg["device"],
    )

    video_path = RAW_DIR / "video_linea_01.mp4"
    if not video_path.exists():
        print(f"[ERROR] No se encontró: {video_path}")
        sys.exit(1)

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[VIDEO] {video_path.name}  {w}×{h} @ {fps:.1f}fps — {total} frames")
    print(f"\n[PROC]  Buscando {MAX_SAVED} frames con detecciones dentro del ROI...\n")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    saved        = []
    frame_number = 0

    while len(saved) < MAX_SAVED:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_frame(frame, roi=roi)

        if detections:
            cats    = [d["category_name"] for d in detections]
            resumen = ", ".join(
                f"{c}×{cats.count(c)}" for c in dict.fromkeys(cats)
            )
            print(f"  frame {frame_number:06d} | det={len(detections):>2} | {resumen}")

            annotated = draw_frame(frame, detections, roi=roi)
            idx       = len(saved) + 1
            out_name  = f"figura3_deteccion_{idx:02d}.jpg"
            out_path  = EXPORTS_DIR / out_name
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(out_path)

        frame_number += 1

    cap.release()

    print(f"\n{'─'*60}")
    print(f"  Frames analizados : {frame_number}")
    print(f"  Imágenes guardadas: {len(saved)}")
    print(f"{'─'*60}")
    for p in saved:
        print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
