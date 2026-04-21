"""
03_comparison.py - compara los 6 operadores de detección de bordes
calcula cpr (porcentaje de pixeles) y métricas de calidad de detección
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


def calc_uniformidad(mascara):
    """
    Calcula uniformidad de la detección (métrica de calidad mejorada)
    Valores altos indican detección más localizada (mejor)
    Valores bajos indican ruido distribuido (peor)

    Se basa en la entropía espacial de los componentes conectados.
    """
    num_etiquetas, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if num_etiquetas <= 1:
        return 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return 0.0

    # Normalizar áreas
    areas_norm = areas / np.sum(areas)

    # Entropy: baja entropía = pocos componentes grandes = mejor
    # Alta entropía = muchos componentes pequeños = peor
    entropy = -np.sum(areas_norm[areas_norm > 0] * np.log2(areas_norm[areas_norm > 0] + 1e-10))

    # Invertir para que mayor = mejor
    max_entropy = np.log2(len(areas))
    uniformidad = (max_entropy - entropy) / max_entropy if max_entropy > 0 else 0

    return max(0, min(1, uniformidad))


def ejecutar():
    """compara operadores y genera graficos"""
    os.makedirs(dir_graficos, exist_ok=True)

    resultados = {op: {"cpr": [], "uniformidad": []} for op in operadores}

    # obtener lista de imagenes
    primer_op_dir = os.path.join(dir_masks, operadores[0])
    if not os.path.exists(primer_op_dir):
        print("  error: no se encontraron mascaras. ejecuta 02_edge_detection.py primero")
        return

    imgs = sorted([f for f in os.listdir(primer_op_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not imgs:
        print("  error: no hay imagenes en las mascaras")
        return

    print(f"  comparando {len(operadores)} operadores en {len(imgs)} imagenes...")

    for nombre in imgs:
        for op in operadores:
            ruta_mascara = os.path.join(dir_masks, op, nombre)
            mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)
            if mascara is None:
                continue
            resultados[op]["cpr"].append(calc_cpr(mascara))
            resultados[op]["uniformidad"].append(calc_uniformidad(mascara))

    # promedios
    resumen = []
    for op in operadores:
        cpr_prom = np.mean(resultados[op]["cpr"]) if resultados[op]["cpr"] else 0
        uniformidad_prom = np.mean(resultados[op]["uniformidad"]) if resultados[op]["uniformidad"] else 0
        resumen.append({
            "operador": op.capitalize(),
            "cpr_promedio": round(cpr_prom, 4),
            "uniformidad_promedio": round(uniformidad_prom, 4)
        })

    df = pd.DataFrame(resumen)
    print("\n  tabla comparativa:")
    print(df.to_string(index=False))

    # guardar csv
    df.to_csv(os.path.join(dir_graficos, "comparacion_operadores.csv"), index=False)

    # grafico comparativo mejorado
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("comparacion de operadores de deteccion de bordes", fontsize=15, fontweight='bold')

    colores = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

    # cpr promedio
    ejes[0].bar(df["operador"], df["cpr_promedio"], color=colores, edgecolor='black', linewidth=1.5)
    ejes[0].set_title("cpr promedio (%)\n(menor = detección más precisa)", fontweight='bold')
    ejes[0].set_ylabel("cpr (%)")
    ejes[0].set_xlabel("operador")
    ejes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df["cpr_promedio"]):
        ejes[0].text(i, v + 0.15, f"{v:.3f}%", ha='center', fontsize=10, fontweight='bold')

    # uniformidad promedio (métrica de calidad)
    ejes[1].bar(df["operador"], df["uniformidad_promedio"], color=colores, edgecolor='black', linewidth=1.5)
    ejes[1].set_title("uniformidad promedio\n(mayor = mejor calidad de detección)", fontweight='bold')
    ejes[1].set_ylabel("uniformidad (0-1)")
    ejes[1].set_xlabel("operador")
    ejes[1].set_ylim(0, 1)
    ejes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df["uniformidad_promedio"]):
        ejes[1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    ruta_out = os.path.join(dir_graficos, "comparacion_operadores.png")
    plt.savefig(ruta_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] grafico guardado en {ruta_out}")

    # determinar mejor operador (criterios: bajo CPR + alta uniformidad)
    # Normalizar para comparación justa
    df["cpr_norm"] = 1 - (df["cpr_promedio"] / df["cpr_promedio"].max())  # Invertir: menor es mejor
    df["score"] = (df["cpr_norm"] + df["uniformidad_promedio"]) / 2

    mejor_idx = df["score"].idxmax()
    mejor = df.loc[mejor_idx, "operador"].lower()
    score_mejor = df.loc[mejor_idx, "score"]

    print(f"\n  mejor operador (score={score_mejor:.4f}): {mejor.upper()}")
    print(f"    - CPR: {df.loc[mejor_idx, 'cpr_promedio']:.4f}%")
    print(f"    - Uniformidad: {df.loc[mejor_idx, 'uniformidad_promedio']:.4f}")

    return mejor


if __name__ == "__main__":
    ejecutar()