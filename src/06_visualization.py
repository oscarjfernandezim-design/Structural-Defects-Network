"""
06_visualization.py
Genera visualizaciones finales del pipeline:
- Histograma de CPR
- Barras de categorías de daño
- Mosaico original | mascara | overlay
- Espectro FFT de la imagen con mayor CPR
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
MASKS_DIR = "results/masks"
GRAPHS_DIR = "results/graphs"
VIS_DIR = "results/visualizations"
SUMMARY_CSV = "results_summary.csv"


def _plot_cpr_histogram(df):
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.hist(df["cpr"], bins=20, color="#1E88E5", edgecolor="white", alpha=0.9)
	ax.axvline(5.0, color="#FBC02D", linestyle="--", linewidth=1.5, label="umbral moderado (5%)")
	ax.axvline(20.0, color="#E53935", linestyle="--", linewidth=1.5, label="umbral severo (20%)")
	ax.set_title("Distribucion de Crack Pixel Ratio (CPR)", fontsize=14, fontweight="bold")
	ax.set_xlabel("CPR (%)")
	ax.set_ylabel("Numero de imagenes")
	ax.legend()
	plt.tight_layout()
	out = os.path.join(GRAPHS_DIR, "01_histograma_cpr.png")
	plt.savefig(out, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"[OK] Histograma CPR guardado en {out}")


def _plot_damage_bars(df):
	counts = df["grado_daño"].value_counts().reindex(["leve", "moderado", "severo"], fill_value=0)
	colors = {"leve": "#43A047", "moderado": "#FBC02D", "severo": "#E53935"}

	fig, ax = plt.subplots(figsize=(8, 5))
	bars = ax.bar(
		counts.index,
		counts.values,
		color=[colors[k] for k in counts.index],
		edgecolor="white",
		width=0.55,
	)

	for bar, val in zip(bars, counts.values):
		ax.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height() + 0.2,
			str(int(val)),
			ha="center",
			va="bottom",
			fontsize=11,
			fontweight="bold",
		)

	ax.set_title("Clasificacion de daño estructural", fontsize=14, fontweight="bold")
	ax.set_xlabel("Grado de daño")
	ax.set_ylabel("Numero de imagenes")
	plt.tight_layout()
	out = os.path.join(GRAPHS_DIR, "02_barras_categorias.png")
	plt.savefig(out, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"[OK] Barras de categorias guardadas en {out}")


def _plot_mosaic(df, best_operator="canny", n=9):
	selected = df.head(n)["imagen"].tolist()
	if not selected:
		print("[WARN] No hay imagenes para construir mosaico.")
		return

	rows = len(selected)
	fig, axes = plt.subplots(rows, 3, figsize=(12, rows * 3.2), squeeze=False)
	fig.suptitle(
		f"Original | Mascara ({best_operator}) | Overlay",
		fontsize=14,
		fontweight="bold",
	)

	for i, name in enumerate(selected):
		orig_path = os.path.join(PROCESSED_DIR, name)
		mask_path = os.path.join(MASKS_DIR, best_operator, name)

		orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

		if orig is None or mask is None:
			axes[i][0].axis("off")
			axes[i][1].axis("off")
			axes[i][2].axis("off")
			continue

		overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
		overlay[mask > 0] = [225, 60, 60]

		row = df[df["imagen"] == name].iloc[0]
		label = f"{name} | CPR: {row['cpr']:.3f}% | {row['grado_daño']}"

		axes[i][0].imshow(orig, cmap="gray")
		axes[i][0].set_title("Original", fontsize=8)
		axes[i][0].axis("off")

		axes[i][1].imshow(mask, cmap="gray")
		axes[i][1].set_title("Mascara", fontsize=8)
		axes[i][1].axis("off")

		axes[i][2].imshow(overlay)
		axes[i][2].set_title(label, fontsize=7)
		axes[i][2].axis("off")

	plt.tight_layout()
	out = os.path.join(VIS_DIR, "03_mosaico_comparacion.png")
	plt.savefig(out, dpi=130, bbox_inches="tight")
	plt.close()
	print(f"[OK] Mosaico de comparacion guardado en {out}")


def _plot_fft_spectrum(df):
	if "cpr" not in df.columns or df.empty:
		return

	worst_idx = df["cpr"].idxmax()
	worst_img = df.loc[worst_idx, "imagen"]
	img_path = os.path.join(PROCESSED_DIR, worst_img)
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	if img is None:
		print(f"[WARN] No se pudo cargar imagen para FFT: {img_path}")
		return

	freq = np.fft.fft2(img)
	freq_shift = np.fft.fftshift(freq)
	magnitude = 20 * np.log(np.abs(freq_shift) + 1)

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	fig.suptitle(f"Analisis FFT - {worst_img}", fontsize=13, fontweight="bold")

	axes[0].imshow(img, cmap="gray")
	axes[0].set_title("Imagen preprocesada")
	axes[0].axis("off")

	axes[1].imshow(magnitude, cmap="inferno")
	axes[1].set_title("Espectro de magnitud")
	axes[1].axis("off")

	plt.tight_layout()
	out = os.path.join(GRAPHS_DIR, "04_fft_spectrum.png")
	plt.savefig(out, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"[OK] Espectro FFT guardado en {out}")


def run(best_operator="canny"):
	"""Genera todas las visualizaciones finales."""
	os.makedirs(GRAPHS_DIR, exist_ok=True)
	os.makedirs(VIS_DIR, exist_ok=True)

	if not os.path.exists(SUMMARY_CSV):
		print(f"[ERROR] No se encontro {SUMMARY_CSV}. Ejecuta metrics antes.")
		return

	df = pd.read_csv(SUMMARY_CSV)
	if df.empty:
		print(f"[ERROR] {SUMMARY_CSV} esta vacio. No hay datos para visualizar.")
		return

	required = {"imagen", "cpr", "grado_daño"}
	missing = [c for c in required if c not in df.columns]
	if missing:
		print(f"[ERROR] Faltan columnas en {SUMMARY_CSV}: {missing}")
		return

	print(f"[INFO] Generando visualizaciones para {len(df)} imagenes...")
	_plot_cpr_histogram(df)
	_plot_damage_bars(df)
	_plot_mosaic(df, best_operator=best_operator)
	_plot_fft_spectrum(df)
	print("[OK] Visualizaciones generadas correctamente.")


if __name__ == "__main__":
	run()
