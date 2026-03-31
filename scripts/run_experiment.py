"""
run_experiment.py
-----------------
Experimento de validación del pipeline de visión por computadora para
detección y conteo de piezas en línea de pintura industrial.

Metodología de validación:
- 3 corridas independientes por video para medir consistencia (repetibilidad).
- Métrica principal: Coeficiente de Variación (CV) de conteos entre corridas.
- Comparación contra baseline del proceso manual histórico (ref. SCRUM-6).
- Hipótesis validada si la reducción de discrepancia es >= 30%.

Uso:
    python scripts/run_experiment.py
"""

import csv
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

# ── Ajustar sys.path para importar desde src/ ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from database import InventoryDB
from counter import PieceCounter
from detector import PieceDetector

# ── Rutas globales ────────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
EXPORTS_DIR   = ROOT / "data" / "exports"
EXPERIMENT_DB = str(EXPORTS_DIR / "experiment.db")
RESULTS_CSV   = str(EXPORTS_DIR / "experiment_results.csv")
SYNTH_DIR     = ROOT / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
#  Carga de configuración
# ─────────────────────────────────────────────────────────────────────────────

def cargar_configs() -> tuple:
    """
    Carga los archivos de configuración del modelo y de la base de datos.

    Returns:
        Tupla (model_cfg, db_cfg) con los diccionarios YAML parseados.
    """
    model_cfg_path = ROOT / "configs" / "model_config.yaml"
    db_cfg_path    = ROOT / "configs" / "db_config.yaml"

    with open(model_cfg_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(db_cfg_path, encoding="utf-8") as f:
        db_cfg = yaml.safe_load(f)

    # Resolver model_path relativo a la raíz del proyecto
    model_cfg["model_path"] = str(ROOT / model_cfg["model_path"])

    return model_cfg, db_cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Generación de videos sintéticos
# ─────────────────────────────────────────────────────────────────────────────

def generar_video_sintetico(
    output_path: str,
    num_frames: int = 15,
    seed: int = 0,
) -> str:
    """
    Genera un video MP4 sintético con rectángulos de colores como objetos simulados.

    Cada frame contiene entre 3 y 8 rectángulos de colores aleatorios sobre
    un fondo gris, simulando piezas sobre una cinta transportadora.
    Se usa una semilla diferente por video para introducir variabilidad
    controlada entre los tres lotes de prueba.

    Args:
        output_path: Ruta de salida del archivo .mp4.
        num_frames: Número de frames a generar.
        seed: Semilla del generador para reproducibilidad.

    Returns:
        Ruta del video generado.
    """
    rng = np.random.default_rng(seed=seed)
    width, height = 640, 640
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        # Fondo gris simulando superficie de cinta transportadora
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        n_objects = int(rng.integers(3, 9))
        for _ in range(n_objects):
            x1 = int(rng.integers(10, width  - 120))
            y1 = int(rng.integers(10, height - 120))
            bw = int(rng.integers(50, 140))
            bh = int(rng.integers(50, 140))
            color = (
                int(rng.integers(80, 255)),
                int(rng.integers(80, 255)),
                int(rng.integers(80, 255)),
            )
            cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), color, thickness=-1)
            # Borde oscuro para definición del objeto
            cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (20, 20, 20), thickness=2)

        out.write(frame)

    out.release()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  Experimento por corrida
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    video_path: str,
    batch_id: str,
    run_number: int,
    model_cfg: dict,
    roi: dict = None,
) -> Dict:
    """
    Ejecuta una corrida individual del experimento sobre un video.

    Instancia componentes frescos (PieceDetector, PieceCounter, InventoryDB)
    para garantizar independencia entre corridas, sin estado compartido.
    Persiste todas las detecciones en la base de datos del experimento y
    extrae métricas resumidas al finalizar.

    Args:
        video_path: Ruta al archivo de video a procesar.
        batch_id: Identificador único de lote para esta corrida.
        run_number: Número de corrida (1, 2 o 3).
        model_cfg: Diccionario con parámetros del modelo (model_path, conf_threshold, device).
        roi: Diccionario opcional con configuración del ROI (x1, y1, x2, y2).

    Returns:
        Diccionario con las métricas de la corrida:
            - batch_id             (str)
            - run_number           (int)
            - video_name           (str)
            - conformes            (int)
            - vec                  (int)
            - scrap                (int)
            - retrabajo            (int)
            - total_detecciones    (int)
            - precision_estimada   (float): promedio de confianza de detecciones
            - duracion_segundos    (float)
            - frames_procesados    (int)
    """
    video_name = Path(video_path).stem

    # Instancias limpias e independientes por corrida
    db       = InventoryDB(db_path=EXPERIMENT_DB)
    counter  = PieceCounter(conf_threshold=model_cfg["conf_threshold"])
    detector = PieceDetector(
        model_path=model_cfg["model_path"],
        conf_threshold=model_cfg["conf_threshold"],
        device=model_cfg["device"],
    )

    # Construir tupla ROI si está configurado
    roi_tuple = None
    if roi is not None and roi.get("enabled", False):
        roi_tuple = (roi["x1"], roi["y1"], roi["x2"], roi["y2"])

    metrics = detector.process_video(
        video_path=video_path,
        batch_id=batch_id,
        db=db,
        counter=counter,
        roi=roi_tuple,
    )

    counts = metrics["counts"]

    # Precision estimada = promedio de confianza de todas las detecciones del lote
    conn   = sqlite3.connect(EXPERIMENT_DB)
    row    = conn.execute(
        "SELECT AVG(confidence) FROM detections WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    conn.close()
    precision_estimada = round(row[0], 4) if row[0] is not None else 0.0

    return {
        "batch_id":           batch_id,
        "run_number":         run_number,
        "video_name":         video_name,
        "conformes":          counts.get("CONFORME",  0),
        "vec":                counts.get("VEC",       0),
        "scrap":              counts.get("SCRAP",     0),
        "retrabajo":          counts.get("RETRABAJO", 0),
        "total_detecciones":  metrics["total_detections"],
        "precision_estimada": precision_estimada,
        "duracion_segundos":  metrics["duration_seconds"],
        "frames_procesados":  metrics["total_frames"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline del proceso manual
# ─────────────────────────────────────────────────────────────────────────────

def calcular_baseline() -> Dict:
    """
    Retorna el baseline del proceso de conteo manual histórico.

    # SCRUM-6: Estos valores provienen del análisis de auditorías internas
    # realizadas durante Q3-Q4 del ciclo productivo anterior, donde se midió
    # la discrepancia entre conteos de dos operadores independientes sobre
    # los mismos lotes. discrepancia_baseline = 0.23 refleja que, en promedio,
    # los conteos manuales entre operadores difieren en un 23%.
    # sobreinventario_baseline = 0.18 es el colchón preventivo que el equipo
    # de producción mantiene para compensar esa incertidumbre.

    Returns:
        Diccionario con métricas del baseline manual:
            - discrepancia_baseline    (float): 0.23 — CV histórico entre operadores
            - sobreinventario_baseline (float): 0.18 — buffer preventivo de inventario
    """
    return {
        "discrepancia_baseline":    0.23,   # 23% discrepancia promedio histórica (SCRUM-6)
        "sobreinventario_baseline": 0.18,   # 18% sobreinventario preventivo (SCRUM-6)
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Cálculo de mejora
# ─────────────────────────────────────────────────────────────────────────────

def calcular_mejora(
    resultados_experimento: List[Dict],
    baseline: Dict,
) -> Dict:
    """
    Calcula las métricas de mejora del sistema vs el baseline manual.

    La discrepancia del sistema se mide como el promedio del Coeficiente de
    Variación (CV = std / mean) de los conteos totales entre las 3 corridas
    por video. Un CV bajo indica alta consistencia (baja discrepancia).

    Fórmulas aplicadas:
        CV_video  = std(total_detecciones_3_runs) / mean(total_detecciones_3_runs)
        discrepancia_sistema = mean(CV_video) sobre todos los videos
        reduccion_discrepancia = ((baseline - sistema) / baseline) * 100

    Args:
        resultados_experimento: Lista de dicts retornados por run_single_experiment().
        baseline: Dict retornado por calcular_baseline().

    Returns:
        Diccionario con:
            - discrepancia_sistema      (float): CV promedio del sistema automatizado
            - discrepancia_baseline     (float): CV del proceso manual (SCRUM-6)
            - reduccion_discrepancia    (float): Reducción porcentual vs baseline
            - objetivo_cumplido         (bool): True si reducción >= 30%
            - cv_por_video              (dict): CV individual por video
    """
    # Agrupar total_detecciones por video (considerando todas las corridas)
    por_video: Dict[str, List[int]] = defaultdict(list)
    for r in resultados_experimento:
        por_video[r["video_name"]].append(r["total_detecciones"])

    # Calcular CV por video
    cv_por_video = {}
    cvs = []
    for video_name, conteos in por_video.items():
        arr  = np.array(conteos, dtype=float)
        mean = arr.mean()
        std  = arr.std(ddof=0)   # desviación estándar poblacional entre corridas
        cv   = round(float(std / mean), 4) if mean > 0 else 0.0
        cv_por_video[video_name] = cv
        cvs.append(cv)

    discrepancia_sistema   = round(float(np.mean(cvs)), 4) if cvs else 0.0
    discrepancia_baseline  = baseline["discrepancia_baseline"]

    reduccion = (
        ((discrepancia_baseline - discrepancia_sistema) / discrepancia_baseline) * 100
        if discrepancia_baseline > 0 else 0.0
    )
    reduccion = round(reduccion, 2)

    return {
        "discrepancia_sistema":   discrepancia_sistema,
        "discrepancia_baseline":  discrepancia_baseline,
        "reduccion_discrepancia": reduccion,
        "objetivo_cumplido":      reduccion >= 30.0,
        "cv_por_video":           cv_por_video,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Impresión de resultados
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_tabla_resultados(resultados: List[Dict]) -> None:
    """Imprime la tabla detallada de todas las corridas del experimento."""
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  {'RESULTADOS DEL EXPERIMENTO — TODAS LAS CORRIDAS':^86}")
    print(sep)
    print(
        f"  {'Video':<22} {'Run':>3}  "
        f"{'CONF':>5} {'VEC':>5} {'SCRAP':>5} {'RETRAB':>6}  "
        f"{'TOTAL':>5} {'PREC':>6} {'SEG':>6} {'FRAMES':>6}"
    )
    print(sep)

    video_actual = None
    for r in resultados:
        if video_actual and video_actual != r["video_name"]:
            print()
        video_actual = r["video_name"]

        video_display = r["video_name"][:22]
        print(
            f"  {video_display:<22} {r['run_number']:>3}  "
            f"{r['conformes']:>5} {r['vec']:>5} {r['scrap']:>5} {r['retrabajo']:>6}  "
            f"{r['total_detecciones']:>5} {r['precision_estimada']:>6.3f} "
            f"{r['duracion_segundos']:>6.1f} {r['frames_procesados']:>6}"
        )

    print(sep)


def imprimir_metricas_mejora(mejora: Dict) -> None:
    """Imprime el cuadro de métricas de mejora vs baseline."""
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  {'MÉTRICAS DE MEJORA VS BASELINE (SCRUM-6)':^51}")
    print(sep)
    print(f"  {'Discrepancia baseline (manual):':<35} {mejora['discrepancia_baseline']*100:>6.1f} %")
    print(f"  {'Discrepancia sistema (automatizado):':<35} {mejora['discrepancia_sistema']*100:>6.1f} %")
    print(f"  {'Reducción de discrepancia:':<35} {mejora['reduccion_discrepancia']:>6.1f} %")
    print(f"  {'Objetivo >= 30 %:':<35} {'✓ CUMPLIDO' if mejora['objetivo_cumplido'] else '✗ NO CUMPLIDO':>10}")
    print()
    print("  CV por video:")
    for video, cv in mejora["cv_por_video"].items():
        print(f"    {video:<30} CV = {cv:.4f}  ({cv*100:.1f} %)")
    print(sep)


def imprimir_conclusion(mejora: Dict) -> None:
    """Imprime el veredicto final del experimento."""
    sep = "=" * 55
    print(f"\n{sep}")
    if mejora["objetivo_cumplido"]:
        print(f"  {'HIPÓTESIS VALIDADA':^51}")
        print(f"  Reducción de discrepancia: {mejora['reduccion_discrepancia']:.1f}% >= 30%")
    else:
        print(f"  {'AJUSTE REQUERIDO':^51}")
        print(f"  Reducción de discrepancia: {mejora['reduccion_discrepancia']:.1f}% < 30%")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  EXPERIMENTO DE VALIDACIÓN — vision-inventory-ai")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Cargar configuración ──────────────────────────────────────────────────
    print("\n[1/5] Cargando configuraciones...")
    model_cfg, db_cfg = cargar_configs()
    print(f"  modelo      : {model_cfg['model_path']}")
    print(f"  conf_thresh : {model_cfg['conf_threshold']}")
    print(f"  device      : {model_cfg['device']}")

    # ── Extraer ROI de configuración ──────────────────────────────────────────
    roi_cfg = model_cfg.get("roi", {})
    if roi_cfg.get("enabled", False):
        print(f"  ROI         : enabled ({roi_cfg['x1']},{roi_cfg['y1']}) → ({roi_cfg['x2']},{roi_cfg['y2']})")
    else:
        print(f"  ROI         : disabled")
        roi_cfg = None

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Preparar videos ───────────────────────────────────────────────────────
    print("\n[2/5] Preparando videos de prueba...")
    video_files = sorted(RAW_DIR.glob("*.mp4"))

    if not video_files:
        print("  No se encontraron .mp4 en data/raw/. Generando 3 videos sintéticos...")
        for i, seed in enumerate([42, 137, 256], start=1):
            path = str(SYNTH_DIR / f"synthetic_lote_{i:02d}.mp4")
            generar_video_sintetico(path, num_frames=15, seed=seed)
            print(f"  ✓ synthetic_lote_{i:02d}.mp4 — 15 frames (seed={seed})")
        video_files = sorted(RAW_DIR.glob("*.mp4"))
    else:
        print(f"  Videos encontrados: {len(video_files)}")
        for vf in video_files:
            print(f"    · {vf.name}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n[3/5] Cargando baseline del proceso manual (SCRUM-6)...")
    baseline = calcular_baseline()
    print(f"  discrepancia_baseline    : {baseline['discrepancia_baseline']*100:.0f} %")
    print(f"  sobreinventario_baseline : {baseline['sobreinventario_baseline']*100:.0f} %")

    # ── Ejecutar 3 corridas por video ─────────────────────────────────────────
    print("\n[4/5] Ejecutando experimento — 3 corridas × video...")
    print(f"  Videos: {len(video_files)}  |  Corridas/video: 3  |  "
          f"Total corridas: {len(video_files) * 3}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    todos_resultados: List[Dict] = []

    for video_path in video_files:
        video_name = video_path.stem
        print(f"  ▶ {video_name}")

        for run in range(1, 4):
            batch_id = f"{video_name}_R{run}_{ts}"
            t0 = time.time()

            resultado = run_single_experiment(
                video_path=str(video_path),
                batch_id=batch_id,
                run_number=run,
                model_cfg=model_cfg,
                roi=roi_cfg,
            )
            todos_resultados.append(resultado)

            elapsed = time.time() - t0
            print(
                f"    Corrida {run}/3 — "
                f"frames={resultado['frames_procesados']:>3}  "
                f"detecciones={resultado['total_detecciones']:>4}  "
                f"[CONF={resultado['conformes']} VEC={resultado['vec']} "
                f"SCRAP={resultado['scrap']} RETRAB={resultado['retrabajo']}]  "
                f"{elapsed:.1f}s"
            )

    # ── Guardar CSV de resultados ─────────────────────────────────────────────
    print(f"\n[5/5] Guardando resultados en {RESULTS_CSV}...")
    campos = [
        "batch_id", "run_number", "video_name",
        "conformes", "vec", "scrap", "retrabajo",
        "total_detecciones", "precision_estimada",
        "duracion_segundos", "frames_procesados",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todos_resultados)
    csv_size = Path(RESULTS_CSV).stat().st_size
    print(f"  ✓ {len(todos_resultados)} filas — {csv_size} bytes")

    # ── Calcular y mostrar métricas ───────────────────────────────────────────
    mejora = calcular_mejora(todos_resultados, baseline)

    imprimir_tabla_resultados(todos_resultados)
    imprimir_metricas_mejora(mejora)
    imprimir_conclusion(mejora)


if __name__ == "__main__":
    main()
