"""
ejecutar.py - script principal para el analisis completo de daño estructural
uso: python ejecutar.py
"""

import importlib
import importlib.util
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Los modulos actuales usan rutas relativas al proyecto.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(SRC_DIR))


def cargar_modulo(nombre, ruta):
    """carga dinamicamente un archivo como modulo"""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar el modulo desde {ruta}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validar_dependencias():
    """Comprueba las dependencias antes de iniciar el pipeline."""
    paquetes = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
    }
    problemas = []
    for modulo, paquete in paquetes.items():
        try:
            importlib.import_module(modulo)
        except ModuleNotFoundError:
            problemas.append(f"{paquete} (no instalado)")
        except ImportError as error:
            problemas.append(f"{paquete} (instalacion incompatible: {error})")

    if problemas:
        print("  [ERR] hay problemas con las dependencias:")
        for problema in problemas:
            print(f"        - {problema}")
        print("  [INFO] instala las dependencias con:")
        print("         python -m pip install -r requirements.txt")
        return False
    return True


def ejecutar_etapa(nombre, modulo, funcion, *args, **kwargs):
    """Ejecuta una etapa y convierte cualquier fallo en un error del pipeline."""
    try:
        resultado = getattr(modulo, funcion)(*args, **kwargs)
    except Exception as error:
        print(f"  [ERR] error en {nombre}: {error}")
        return None, False
    return resultado, True


def main():
    if not validar_dependencias():
        return 1

    total_pasos = 6

    print("=" * 70)
    print("  PIPELINE DE ANÁLISIS DE DAÑO ESTRUCTURAL".center(70))
    print("  Detección automatizada de grietas y daño en infraestructura".center(70))
    print("=" * 70)

    # paso 1: preprocesamiento
    print(f"\n[1/{total_pasos}] preprocesando imagenes...")
    try:
        prep = cargar_modulo("prep", SRC_DIR / "01_preprocessing.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] error cargando preprocesamiento: {error}")
        return 1
    resultado, correcto = ejecutar_etapa("preprocesamiento", prep, "ejecutar")
    if not correcto or not resultado:
        print("  [ERR] el preprocesamiento no produjo resultados")
        return 1

    # paso 2: deteccion de bordes
    print(f"\n[2/{total_pasos}] detectando bordes con 6 operadores...")
    try:
        bordes = cargar_modulo("bordes", SRC_DIR / "02_edge_detection.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] error cargando deteccion de bordes: {error}")
        return 1
    _, correcto = ejecutar_etapa("deteccion de bordes", bordes, "ejecutar")
    if not correcto:
        return 1

    # paso 3: comparacion de operadores
    print(f"\n[3/{total_pasos}] comparando operadores...")
    try:
        comp = cargar_modulo("comp", SRC_DIR / "03_fft_filter.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] error cargando comparacion: {error}")
        return 1
    mejor_op, correcto = ejecutar_etapa("comparacion", comp, "ejecutar")
    if not correcto:
        return 1
    if not mejor_op:
        mejor_op = "canny"
        print(f"  [!] usando operador por defecto: {mejor_op}")

    # Cuando hay etiquetas, la selección debe usar la evaluación calibrada,
    # no el score visual de CPR/uniformidad.
    annotation_dir = PROJECT_ROOT / "data" / "annotated"
    if (annotation_dir / "Cracked").is_dir() and (annotation_dir / "No-Cracked").is_dir():
        try:
            evaluator = cargar_modulo("evaluator", SRC_DIR / "07_evaluate.py")
            _, summary = evaluator.evaluate_folder_labels(
                annotation_dir=annotation_dir,
                prediction_dir=PROJECT_ROOT / "results" / "masks",
            )
            if not summary.empty and summary["f1"].notna().any():
                mejor_op = str(summary.iloc[0]["operador"])
                print(
                    f"  [OK] operador seleccionado por F1 en test etiquetado: "
                    f"{mejor_op.upper()}"
                )
        except (ImportError, OSError, ValueError) as error:
            print(f"  [!] no se pudo usar la evaluación etiquetada: {error}")

    # paso 4: calculo de metricas
    print(f"\n[4/{total_pasos}] calculando metricas con {mejor_op.upper()}...")
    try:
        metr = cargar_modulo("metr", SRC_DIR / "04_comparison.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] error cargando metricas: {error}")
        return 1
    df_metricas, correcto = ejecutar_etapa(
        "metricas", metr, "ejecutar", mejor_op=mejor_op
    )
    if not correcto or df_metricas is None:
        print("  [ERR] no se calcularon metricas")
        return 1

    # paso 5: visualizacion
    print(f"\n[5/{total_pasos}] generando visualizaciones...")
    try:
        viz = cargar_modulo("viz", SRC_DIR / "06_visualization.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] error cargando visualizacion: {error}")
        return 1
    _, correcto = ejecutar_etapa(
        "visualizacion", viz, "ejecutar", mejor_op=mejor_op
    )
    if not correcto:
        return 1

    # paso 6: resumen final
    print(f"\n[6/{total_pasos}] generando reporte final...")
    print("\n" + "=" * 70)
    print("  [OK] ANÁLISIS COMPLETADO".center(70))
    print("=" * 70)
    print(f"   Resultados principales:")
    print(f"     - Operador seleccionado: {mejor_op.upper()}")
    print(f"     - Imágenes procesadas: {len(df_metricas)}")
    print(f"     - CPR promedio: {df_metricas['cpr'].mean():.3f}%")
    print(f"     - Rango CPR: {df_metricas['cpr'].min():.3f}% - {df_metricas['cpr'].max():.3f}%")
    print(f"\n   Archivos generados:")
    print(f"     - results_summary.csv")
    print(f"     - results/graphs/comparacion_operadores.png")
    print(f"     - results/visualizations/03_mosaico_comparacion.png")
    print(f"     - results/graphs/04_fft_spectrum.png")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
