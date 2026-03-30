"""
main.py
-------
Punto de entrada principal del sistema de visión por computadora
para detección y conteo de piezas en línea de pintura industrial.

Uso:
    python main.py                          # procesa todos los .mp4 en data/raw/
    python main.py --video data/raw/lote1.mp4
    python main.py --video data/raw/lote1.mp4 --batch LOTE_2026_001
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Ajustar el path para importar desde src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database import InventoryDB
from counter import PieceCounter
from detector import PieceDetector


def load_config(config_path: str) -> dict:
    """
    Carga un archivo de configuración YAML.

    Args:
        config_path: Ruta al archivo .yaml.

    Returns:
        Diccionario con la configuración cargada.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_batch_id(video_path: str, prefix: str = "LOTE_") -> str:
    """
    Genera un ID de lote único basado en la fecha y el nombre del video.

    Args:
        video_path: Ruta al archivo de video.
        prefix: Prefijo del identificador de lote.

    Returns:
        Cadena con formato PREFIJO_YYYYMMDD_HHMMSS_nombre_archivo.
    """
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}_{video_name}"


def print_summary_table(results: list) -> None:
    """
    Imprime una tabla resumen de los resultados en consola.

    Args:
        results: Lista de diccionarios de métricas retornados por process_video().
    """
    if not results:
        print("\n[INFO] No se procesaron videos.")
        return

    print("\n" + "=" * 80)
    print(f"{'RESUMEN DE PROCESAMIENTO':^80}")
    print("=" * 80)
    header = f"{'Lote':<30} {'Frames':>8} {'Detec.':>8} {'CONF':>6} {'VEC':>6} {'SCRAP':>6} {'RETRAB':>7} {'Seg.':>6}"
    print(header)
    print("-" * 80)

    for r in results:
        counts = r.get("counts", {})
        row = (
            f"{r['batch_id']:<30} "
            f"{r['total_frames']:>8} "
            f"{r['total_detections']:>8} "
            f"{counts.get('CONFORME', 0):>6} "
            f"{counts.get('VEC', 0):>6} "
            f"{counts.get('SCRAP', 0):>6} "
            f"{counts.get('RETRABAJO', 0):>7} "
            f"{r['duration_seconds']:>6.1f}"
        )
        print(row)

    print("=" * 80)
    total_frames = sum(r["total_frames"] for r in results)
    total_dets = sum(r["total_detections"] for r in results)
    print(f"{'TOTAL':<30} {total_frames:>8} {total_dets:>8}")
    print("=" * 80 + "\n")


def main():
    """
    Función principal del pipeline de detección y conteo.

    Flujo de ejecución:
    1. Parsea argumentos de línea de comandos.
    2. Carga configuraciones de modelo y base de datos desde configs/.
    3. Instancia InventoryDB, PieceCounter y PieceDetector.
    4. Procesa uno o todos los videos .mp4 en data/raw/.
    5. Exporta CSV de detecciones por cada video procesado.
    6. Imprime tabla resumen en consola.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de visión para detección y conteo de piezas industriales."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Ruta a un video específico (.mp4). Si no se indica, procesa todos en data/raw/.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="ID de lote personalizado (aplica solo si se usa --video).",
    )
    args = parser.parse_args()

    # Cargar configuraciones
    base_dir = Path(__file__).parent
    model_cfg = load_config(base_dir / "configs" / "model_config.yaml")
    db_cfg = load_config(base_dir / "configs" / "db_config.yaml")

    # Instanciar componentes del pipeline
    db = InventoryDB(db_path=db_cfg["db_path"])
    counter = PieceCounter(conf_threshold=model_cfg["conf_threshold"])
    detector = PieceDetector(
        model_path=model_cfg["model_path"],
        conf_threshold=model_cfg["conf_threshold"],
        device=model_cfg["device"],
    )

    # Determinar lista de videos a procesar
    if args.video:
        video_files = [Path(args.video)]
    else:
        raw_dir = base_dir / "data" / "raw"
        video_files = sorted(raw_dir.glob("*.mp4"))

    if not video_files:
        print("[AVISO] No se encontraron videos para procesar.")
        return

    exports_path = Path(db_cfg["exports_path"])
    exports_path.mkdir(parents=True, exist_ok=True)
    batch_prefix = db_cfg.get("batch_prefix", "LOTE_")

    all_results = []

    for video_path in video_files:
        # Generar o usar el batch_id proporcionado
        if args.batch and args.video:
            batch_id = args.batch
        else:
            batch_id = generate_batch_id(str(video_path), prefix=batch_prefix)

        print(f"\n[INFO] Procesando: {video_path.name} → Lote: {batch_id}")

        try:
            metrics = detector.process_video(
                video_path=str(video_path),
                batch_id=batch_id,
                db=db,
                counter=counter,
            )
            all_results.append(metrics)

            # Exportar CSV de detecciones de este lote
            csv_path = exports_path / f"{batch_id}.csv"
            db.export_csv(batch_id=batch_id, output_path=str(csv_path))
            print(f"[INFO] CSV exportado: {csv_path}")

        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[ERROR] No se pudo procesar {video_path.name}: {exc}")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
