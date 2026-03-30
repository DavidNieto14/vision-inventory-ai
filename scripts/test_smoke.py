"""
test_smoke.py
-------------
Prueba de humo (smoke test) del pipeline completo de visión por computadora.

Verifica que los tres componentes principales (PieceDetector, PieceCounter,
InventoryDB) se instancian, se comunican y persisten datos correctamente
usando una imagen sintética generada con NumPy, sin necesidad de video real.

Uso:
    python scripts/test_smoke.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np

# ── Ajustar sys.path para importar desde src/ ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from database import InventoryDB
from counter import PieceCounter
from detector import PieceDetector

SEPARATOR = "─" * 60
MODEL_PATH = str(ROOT / "models" / "yolov8n.pt")
DB_PATH    = str(ROOT / "data" / "exports" / "inventory.db")
CSV_PATH   = str(ROOT / "data" / "exports" / "smoke_test.csv")
BATCH_ID   = "SMOKE_TEST_001"


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def main() -> None:
    print("\n" + "=" * 60)
    print("  SMOKE TEST — vision-inventory-ai pipeline")
    print("=" * 60)

    # ── 1. PieceDetector ──────────────────────────────────────────────────────
    section("1 · Cargando PieceDetector (yolov8n.pt)")
    detector = PieceDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.25,   # umbral bajo para maximizar detecciones en imagen sintética
        device="cpu",
    )
    print(f"  modelo   : {detector.model_path}")
    print(f"  conf     : {detector.conf_threshold}")
    print(f"  device   : {detector.device}")

    # ── 2. Imagen sintética ───────────────────────────────────────────────────
    section("2 · Generando imagen sintética 640×640×3 (valores aleatorios)")
    rng = np.random.default_rng(seed=42)
    synthetic_frame = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
    print(f"  shape    : {synthetic_frame.shape}")
    print(f"  dtype    : {synthetic_frame.dtype}")
    print(f"  min/max  : {synthetic_frame.min()} / {synthetic_frame.max()}")

    # ── 3. Inferencia con detect_frame() ─────────────────────────────────────
    section("3 · Ejecutando detect_frame() sobre imagen sintética")
    detections = detector.detect_frame(synthetic_frame)
    print(f"  detecciones encontradas: {len(detections)}")
    if detections:
        print("  Resultados:")
        for i, det in enumerate(detections, 1):
            print(
                f"    [{i}] class_id={det['class_id']:>2}  "
                f"category={det['category_name']:<10}  "
                f"conf={det['confidence']:.4f}  "
                f"bbox={det['bbox']}"
            )
    else:
        print("  (sin detecciones — esperado con imagen de ruido puro)")

    # ── 4. PieceCounter ───────────────────────────────────────────────────────
    section("4 · Actualizando PieceCounter con las detecciones obtenidas")
    counter = PieceCounter(conf_threshold=0.25)
    counter.update(detections)
    counts = counter.get_counts()
    print(f"  conf_threshold : {counter.conf_threshold}")
    print("  conteos por categoría:")
    for cat, n in counts.items():
        bar = "█" * n if n else "·"
        print(f"    {cat:<12}: {n:>4}  {bar}")

    # ── 5. Detección sintética forzada para poblar la BD ──────────────────────
    section("5 · Preparando detecciones sintéticas para persistencia")
    synthetic_detections = [
        {"class_id": 0, "category_name": "CONFORME",  "confidence": 0.91, "bbox": [10,  10,  80,  80 ]},
        {"class_id": 1, "category_name": "VEC",        "confidence": 0.76, "bbox": [100, 50,  180, 130]},
        {"class_id": 2, "category_name": "SCRAP",      "confidence": 0.83, "bbox": [200, 200, 300, 300]},
        {"class_id": 3, "category_name": "RETRABAJO",  "confidence": 0.67, "bbox": [320, 10,  400, 90 ]},
        {"class_id": 0, "category_name": "CONFORME",  "confidence": 0.95, "bbox": [410, 410, 500, 500]},
    ]
    counter_db = PieceCounter(conf_threshold=0.25)
    counter_db.update(synthetic_detections)
    print(f"  detecciones preparadas : {len(synthetic_detections)}")
    print(f"  conteos antes de BD    : {counter_db.get_counts()}")

    # ── 6. InventoryDB — tablas e inserciones ─────────────────────────────────
    section("6 · Instanciando InventoryDB e insertando detecciones")
    db = InventoryDB(db_path=DB_PATH)
    print(f"  db_path  : {db.db_path}")

    for frame_num, det in enumerate(synthetic_detections):
        db.insert_detection(
            batch_id=BATCH_ID,
            category=det["category_name"],
            confidence=det["confidence"],
            frame_number=frame_num,
        )
    print(f"  filas insertadas en 'detections': {len(synthetic_detections)}")

    db.update_summary(batch_id=BATCH_ID, counts_dict=counter_db.get_counts())
    print(f"  resumen actualizado en 'inventory_summary'")

    summary_df = db.get_summary(BATCH_ID)
    print("\n  DataFrame de resumen:")
    print(summary_df.to_string(index=False))

    # ── 7. Exportar CSV ───────────────────────────────────────────────────────
    section("7 · Exportando CSV a data/exports/smoke_test.csv")
    db.export_csv(batch_id=BATCH_ID, output_path=CSV_PATH)
    csv_size = Path(CSV_PATH).stat().st_size
    print(f"  archivo  : {CSV_PATH}")
    print(f"  tamaño   : {csv_size} bytes")

    # Mostrar primeras líneas del CSV
    with open(CSV_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  filas    : {len(lines) - 1} (+ 1 header)")
    print("\n  Contenido del CSV:")
    for line in lines:
        print("    " + line.rstrip())

    # ── 8. Resultado final ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 60)
        print("  SMOKE TEST FAILED")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)
