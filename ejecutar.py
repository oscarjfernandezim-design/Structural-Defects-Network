"""
ejecutar.py - script principal para el analisis completo
uso: python ejecutar.py
"""

import os
import sys
import importlib.util

# agregar directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def cargar_modulo(nombre, ruta):
    """carga dinamicamente un archivo como modulo"""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    dir_src = os.path.join(os.path.dirname(__file__), 'src')

    print("=" * 60)
    print("  analisis de daño estructural")
    print("=" * 60)

    # paso 1: preprocesamiento
    print("\n[1/6] preprocesando imagenes...")
    try:
        cargar_modulo("prep", os.path.join(dir_src, "01_preprocessing.py")).ejecutar()
    except Exception as e:
        print(f"  error en preprocesamiento: {e}")
        return

    # paso 2: deteccion de bordes
    print("\n[2/5] detectando bordes con 6 operadores...")
    try:
        cargar_modulo("bordes", os.path.join(dir_src, "02_edge_detection.py")).ejecutar()
    except Exception as e:
        print(f"  error en deteccion de bordes: {e}")
        return

    # paso 3: comparacion
    print("\n[3/5] comparando operadores...")
    try:
        mejor_op = cargar_modulo("comp", os.path.join(dir_src, "03_fft_filter.py")).ejecutar()
        if not mejor_op:
            mejor_op = "canny"
            print(f"  (usando operador por defecto: {mejor_op})")
    except Exception as e:
        print(f"  error en comparacion: {e}")
        return

    # paso 4: metricas
    print(f"\n[4/5] calculando metricas con {mejor_op}...")
    try:
        cargar_modulo("metr", os.path.join(dir_src, "04_comparison.py")).ejecutar(mejor_op=mejor_op)
    except Exception as e:
        print(f"  error en metricas: {e}")
        return

    # paso 5: visualizacion
    print(f"\n[5/5] generando graficos...")
    try:
        cargar_modulo("viz", os.path.join(dir_src, "06_visualization.py")).run(best_operator=mejor_op)
    except Exception as e:
        print(f"  error en visualizacion: {e}")
        return

    print("\n" + "=" * 60)
    print("  ✓ analisis completado")
    print("  resultados: results/")
    print("  tabla: results_summary.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
