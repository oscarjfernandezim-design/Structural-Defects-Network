"""
05_visualizacion.py - genera todos los graficos finales
histograma cpr, barras de categorias, mosaico, espectro fft
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir_proc = "data/processed"
dir_masks = "results/masks"
dir_graficos = "results/graphs"
dir_viz = "results/visualizations"
csv_path = "results_summary.csv"


def grafico_histogram_cpr(df):
    """histograma del crack pixel ratio"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["cpr"], bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(5.0, color="#FFC107", linestyle="--", linewidth=1.5, label="umbral moderado (5%)")
    ax.axvline(20.0, color="#F44336", linestyle="--", linewidth=1.5, label="umbral severo (20%)")
    ax.set_title("distribucion del crack pixel ratio (cpr)", fontsize=14, fontweight="bold")
    ax.set_xlabel("cpr (%)")
    ax.set_ylabel("numero de imagenes")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(dir_graficos, "01_histograma_cpr.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def grafico_barras_daño(df):
    """barras de categorias de daño"""
    conteos = df["grado_daño"].value_counts().reindex(["leve", "moderado", "severo"], fill_value=0)
    colores = {"leve": "#4CAF50", "moderado": "#FFC107", "severo": "#F44336"}

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(conteos.index, conteos.values,
                   color=[colores[k] for k in conteos.index],
                   edgecolor="white", width=0.5)

    for barra, val in zip(barras, conteos.values):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("clasificacion de daño estructural", fontsize=14, fontweight="bold")
    ax.set_ylabel("numero de imagenes")
    ax.set_xlabel("grado de daño")
    plt.tight_layout()
    out = os.path.join(dir_graficos, "02_barras_categorias.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def grafico_mosaico(df, mejor_op="canny", n=9):
    """mosaico: original | mascara | overlay"""
    imgs_lista = df.head(n)["imagen"].tolist()
    cols = 3
    filas = len(imgs_lista)

    fig, ejes = plt.subplots(filas, 3, figsize=(12, filas * 4))
    fig.suptitle(f"original | mascara ({mejor_op}) | overlay", fontsize=14, fontweight="bold")

    for i, nombre in enumerate(imgs_lista):
        ruta_orig = os.path.join(dir_proc, nombre)
        ruta_mascara = os.path.join(dir_masks, mejor_op, nombre)

        orig = cv2.imread(ruta_orig, cv2.IMREAD_GRAYSCALE)
        mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)

        if orig is None or mascara is None:
            continue

        # overlay: colorear grietas en rojo
        orig_color = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        overlay = orig_color.copy()
        overlay[mascara > 0] = [220, 50, 50]

        fila = df[df["imagen"] == nombre].iloc[0]
        etiqueta = f"{nombre}\ncpr: {fila['cpr']:.3f}% | {fila['grado_daño']}"

        ejes[i][0].imshow(orig, cmap="gray")
        ejes[i][0].set_title("original", fontsize=8)
        ejes[i][0].axis("off")

        ejes[i][1].imshow(mascara, cmap="gray")
        ejes[i][1].set_title("mascara", fontsize=8)
        ejes[i][1].axis("off")

        ejes[i][2].imshow(overlay)
        ejes[i][2].set_title(etiqueta, fontsize=7)
        ejes[i][2].axis("off")

    plt.tight_layout()
    out = os.path.join(dir_viz, "03_mosaico_comparacion.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def grafico_fft(df):
    """espectro fft de la imagen con mayor daño"""
    # elegir imagen con mayor cpr
    peor_idx = df["cpr"].idxmax()
    peor = df.loc[peor_idx, "imagen"]
    ruta_img = os.path.join(dir_proc, peor)
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return

    # fft
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitud = 20 * np.log(np.abs(fshift) + 1)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"analisis fft - {peor}", fontsize=13, fontweight="bold")

    ejes[0].imshow(img, cmap="gray")
    ejes[0].set_title("imagen original (preprocesada)")
    ejes[0].axis("off")

    ejes[1].imshow(magnitud, cmap="inferno")
    ejes[1].set_title("espectro de magnitud fft")
    ejes[1].axis("off")

    plt.tight_layout()
    out = os.path.join(dir_graficos, "04_fft_spectrum.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def ejecutar(mejor_op="canny"):
    """genera todas las visualizaciones"""
    os.makedirs(dir_graficos, exist_ok=True)
    os.makedirs(dir_viz, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"  error: no encontrado {csv_path}. ejecuta 04_metricas.py primero")
        return

    df = pd.read_csv(csv_path)
    print(f"  generando visualizaciones para {len(df)} imagenes...")

    grafico_histogram_cpr(df)
    grafico_barras_daño(df)
    grafico_mosaico(df, mejor_op=mejor_op)
    grafico_fft(df)

    print(f"\n  ✓ todas las visualizaciones generadas")


if __name__ == "__main__":
    ejecutar()