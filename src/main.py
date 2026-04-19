"""
main.py - analisis de daño en estructuras
ejecuta el pipeline completo
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

    print("=" * 60)
    print("  analisis de daño estructural")
    print("=" * 60)

    # paso 1: preprocesamiento
    print("\n[1/5] preprocesando imagenes...")
    cargar_modulo("prep", os.path.join(base, "01_preprocessing.py")).ejecutar()

    # paso 2: deteccion de bordes
    print("\n[2/5] detectando bordes con 6 operadores...")
    cargar_modulo("bordes", os.path.join(base, "02_edge_detection.py")).ejecutar()

    # paso 3: comparacion
    print("\n[3/5] comparando operadores...")
    mejor_op = cargar_modulo("comp", os.path.join(base, "03_fft_filter.py")).ejecutar()
    if not mejor_op:
        mejor_op = "canny"
        print(f"  (usando operador por defecto: {mejor_op})")

    # paso 4: metricas
    print(f"\n[4/5] calculando metricas con {mejor_op}...")
    cargar_modulo("metr", os.path.join(base, "04_comparison.py")).ejecutar(mejor_op=mejor_op)

    # paso 5: visualizacion
    print(f"\n[5/5] generando graficos...")
    cargar_modulo("viz", os.path.join(base, "06_visualization.py")).run(best_operator=mejor_op)

    print("\n" + "=" * 60)
    print("  ✓ analisis completado")
    print("  resultados: results/")
    print("  tabla: results_summary.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()