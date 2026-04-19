"""
03_comparacion.py - compara los 6 operadores
calcula cpr (porcentaje de pixeles) y snr (relacion señal/ruido)
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir_masks = "results/masks"
dir_graficos = "results/graphs"
operadores = ["canny", "laplaciano", "sobel", "prewitt", "roberts", "fft"]


def calc_cpr(mascara):
    """porcentaje de pixeles blancos respecto al total"""
    total = mascara.size
    grieta = np.count_nonzero(mascara)
    return (grieta / total) * 100


def calc_snr(mascara):
    """
    snr estimado: relacion señal/ruido basado en componentes conectados
    componentes grandes = señal, componentes pequeños = ruido
    """
    num_etiquetas, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if num_etiquetas <= 1:
        return 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return 0.0

    # señal = area del componente mas grande
    senal = np.max(areas)
    # ruido = suma de areas pequeñas
    ruido = np.sum(areas[areas < 50]) + 1
    return senal / ruido


def ejecutar():
    """compara operadores y genera graficos"""
    os.makedirs(dir_graficos, exist_ok=True)

    resultados = {op: {"cpr": [], "snr": []} for op in operadores}

    # obtener lista de imagenes
    primer_op_dir = os.path.join(dir_masks, operadores[0])
    if not os.path.exists(primer_op_dir):
        print("  error: no se encontraron mascaras. ejecuta 02_bordes.py primero")
        return

    imgs = [f for f in os.listdir(primer_op_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  comparando operadores en {len(imgs)} imagenes...")

    for nombre in imgs:
        for op in operadores:
            ruta_mascara = os.path.join(dir_masks, op, nombre)
            mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)
            if mascara is None:
                continue
            resultados[op]["cpr"].append(calc_cpr(mascara))
            resultados[op]["snr"].append(calc_snr(mascara))

    # promedios
    resumen = []
    for op in operadores:
        cpr_prom = np.mean(resultados[op]["cpr"]) if resultados[op]["cpr"] else 0
        snr_prom = np.mean(resultados[op]["snr"]) if resultados[op]["snr"] else 0
        resumen.append({
            "operador": op.capitalize(),
            "cpr_promedio": round(cpr_prom, 4),
            "snr_promedio": round(snr_prom, 2)
        })

    df = pd.DataFrame(resumen)
    print("\n  tabla comparativa:")
    print(df.to_string(index=False))

    # guardar csv
    df.to_csv(os.path.join(dir_graficos, "comparacion_operadores.csv"), index=False)

    # grafico comparativo
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("comparacion de operadores de deteccion de bordes", fontsize=15, fontweight='bold')

    colores = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

    # cpr promedio
    ejes[0].bar(df["operador"], df["cpr_promedio"], color=colores)
    ejes[0].set_title("cpr promedio (%)\n(mayor = detecta mas pixeles)")
    ejes[0].set_ylabel("cpr (%)")
    ejes[0].set_xlabel("operador")
    for i, v in enumerate(df["cpr_promedio"]):
        ejes[0].text(i, v + 0.01, f"{v:.3f}%", ha='center', fontsize=9)

    # snr promedio
    ejes[1].bar(df["operador"], df["snr_promedio"], color=colores)
    ejes[1].set_title("snr promedio\n(mayor = mejor relacion señal/ruido)")
    ejes[1].set_ylabel("snr")
    ejes[1].set_xlabel("operador")
    for i, v in enumerate(df["snr_promedio"]):
        ejes[1].text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)

    plt.tight_layout()
    ruta_out = os.path.join(dir_graficos, "comparacion_operadores.png")
    plt.savefig(ruta_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ grafico guardado en {ruta_out}")

    # determinar mejor operador (mayor snr)
    mejor = df.loc[df["snr_promedio"].idxmax(), "operador"].lower()
    print(f"\n  mejor operador segun snr: {mejor.upper()}")
    return mejor


if __name__ == "__main__":
    ejecutar()