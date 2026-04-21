"""
ejecutar.py - script principal para el analisis completo de daño estructural
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
    total_pasos = 6

    print("=" * 70)
    print("  PIPELINE DE ANÁLISIS DE DAÑO ESTRUCTURAL".center(70))
    print("  Detección automatizada de grietas y daño en infraestructura".center(70))
    print("=" * 70)

    # paso 1: preprocesamiento
    print(f"\n[1/{total_pasos}] preprocesando imagenes...")
    try:
        prep = cargar_modulo("prep", os.path.join(dir_src, "01_preprocessing.py"))
        if not prep.ejecutar():
            print("  [ERR] error: preprocesamiento falló")
            return
    except Exception as e:
        print(f"  [ERR] error en preprocesamiento: {e}")
        return

    # paso 2: deteccion de bordes
    print(f"\n[2/{total_pasos}] detectando bordes con 6 operadores...")
    try:
        bordes = cargar_modulo("bordes", os.path.join(dir_src, "02_edge_detection.py"))
        bordes.ejecutar()
    except Exception as e:
        print(f"  [ERR] error en deteccion de bordes: {e}")
        return

    # paso 3: comparacion de operadores
    print(f"\n[3/{total_pasos}] comparando operadores...")
    try:
        comp = cargar_modulo("comp", os.path.join(dir_src, "03_fft_filter.py"))
        mejor_op = comp.ejecutar()
        if not mejor_op:
            mejor_op = "canny"
            print(f"  [!] usando operador por defecto: {mejor_op}")
    except Exception as e:
        print(f"  [ERR] error en comparacion: {e}")
        return

    # paso 4: calculo de metricas
    print(f"\n[4/{total_pasos}] calculando metricas con {mejor_op.upper()}...")
    try:
        metr = cargar_modulo("metr", os.path.join(dir_src, "04_comparison.py"))
        df_metricas = metr.ejecutar(mejor_op=mejor_op)
        if df_metricas is None:
            print(f"  [ERR] error: no se calcularon metricas")
            return
    except Exception as e:
        print(f"  [ERR] error en metricas: {e}")
        return

    # paso 5: visualizacion
    print(f"\n[5/{total_pasos}] generando visualizaciones...")
    try:
        viz = cargar_modulo("viz", os.path.join(dir_src, "06_visualization.py"))
        viz.ejecutar(mejor_op=mejor_op)
    except Exception as e:
        print(f"  [ERR] error en visualizacion: {e}")
        return

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


if __name__ == "__main__":
    main()
