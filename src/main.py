"""
main.py - punto de entrada alternativo para el análisis de daño estructural
(usar ejecutar.py en su lugar para mejor manejo de errores)
"""

import os
import sys
import importlib.util

def cargar_modulo(nombre, ruta):
    """carga dinamicamente un archivo .py como modulo"""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    total_pasos = 6

    print("=" * 70)
    print("  PIPELINE DE ANÁLISIS DE DAÑO ESTRUCTURAL".center(70))
    print("=" * 70)

    # paso 1: preprocesamiento
    print(f"\n[1/{total_pasos}] preprocesando imagenes...")
    cargar_modulo("prep", os.path.join(base, "01_preprocessing.py")).ejecutar()

    # paso 2: deteccion de bordes
    print(f"\n[2/{total_pasos}] detectando bordes con 6 operadores...")
    cargar_modulo("bordes", os.path.join(base, "02_edge_detection.py")).ejecutar()

    # paso 3: comparacion
    print(f"\n[3/{total_pasos}] comparando operadores...")
    mejor_op = cargar_modulo("comp", os.path.join(base, "03_fft_filter.py")).ejecutar()
    if not mejor_op:
        mejor_op = "canny"
        print(f"  [!] usando operador por defecto: {mejor_op}")

    # paso 4: metricas
    print(f"\n[4/{total_pasos}] calculando metricas con {mejor_op.upper()}...")
    cargar_modulo("metr", os.path.join(base, "04_comparison.py")).ejecutar(mejor_op=mejor_op)

    # paso 5: visualizacion
    print(f"\n[5/{total_pasos}] generando visualizaciones...")
    cargar_modulo("viz", os.path.join(base, "06_visualization.py")).ejecutar(mejor_op=mejor_op)

    # paso 6: resumen
    print(f"\n[6/{total_pasos}] completado")
    print("\n" + "=" * 70)
    print("  [OK] ANÁLISIS COMPLETADO".center(70))
    print("  resultados: results/")
    print("  tabla: results_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()