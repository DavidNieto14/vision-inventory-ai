"""
demo_visual.py
--------------
Demo de visualización del pipeline de detección con soporte de ROI.

Procesa video_linea_03.mp4 con el ROI definido en model_config.yaml,
dibuja bounding boxes solo de detecciones dentro del ROI y guarda
5 frames con detecciones en data/exports/roi_frame_*.jpg

Uso:
    python scripts/demo_visual.py
"""

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import PieceDetector

RAW_DIR     = ROOT / "data" / "raw"
EXPORTS_DIR = ROOT / "data" / "exports"
MAX_SAVED   = 5


def cargar_model_cfg() -> dict:
    cfg_path = ROOT / "configs" / "model_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model_path"] = str(ROOT / cfg["model_path"])
    return cfg


def main() -> None:
    print("\n" + "=" * 60)
    print("  DEMO VISUAL con ROI — vision-inventory-ai")
    print("=" * 60)

    cfg = cargar_model_cfg()

    # ── Leer ROI desde config ─────────────────────────────────────
    roi_cfg = cfg.get("roi", {})
    roi = None
    if roi_cfg.get("enabled", False):
        roi = (roi_cfg["x1"], roi_cfg["y1"], roi_cfg["x2"], roi_cfg["y2"])
        print(f"\n[ROI]  enabled=true  zona=({roi[0]},{roi[1]}) → ({roi[2]},{roi[3]})")
    else:
        print("\n[ROI]  disabled")

    print(f"[INIT] modelo : {cfg['model_path']}")
    print(f"[INIT] conf   : {cfg['conf_threshold']}")
    print(f"[INIT] device : {cfg['device']}")

    detector = PieceDetector(
        model_path=cfg["model_path"],
        conf_threshold=cfg["conf_threshold"],
        device=cfg["device"],
    )

    # ── Video ─────────────────────────────────────────────────────
    video_path = RAW_DIR / "video_linea_03.mp4"
    if not video_path.exists():
        print(f"[ERROR] No se encontró: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[VIDEO] {video_path.name}  ({total} frames, {fps:.1f} fps)")
    print(f"\n[PROC]  Buscando {MAX_SAVED} frames con detecciones dentro del ROI...\n")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    frame_number = 0

    while len(saved) < MAX_SAVED:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_frame(frame, roi=roi)

        if detections:
            cats = [d["category_name"] for d in detections]
            resumen = ", ".join(
                f"{c}×{cats.count(c)}" for c in dict.fromkeys(cats)
            )
            print(f"  frame {frame_number:06d} | det={len(detections):>2} | {resumen}")

            annotated = detector.visualize_detections(frame, detections, roi=roi)
            out_name  = f"roi_frame_{frame_number:06d}.jpg"
            out_path  = EXPORTS_DIR / out_name
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
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
