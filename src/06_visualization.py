"""
06_visualization.py
Generate final pipeline visualizations:
- CPR histogram
- Descriptive severity bar chart
- Original | mask | overlay mosaic
- FFT spectrum for the image with the highest CPR
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
	ax.axvline(5.0, color="#FBC02D", linestyle="--", linewidth=1.5, label="moderate threshold (5%)")
	ax.axvline(20.0, color="#E53935", linestyle="--", linewidth=1.5, label="high threshold (20%)")
	ax.set_title("Crack Pixel Ratio (CPR) distribution", fontsize=14, fontweight="bold")
	ax.set_xlabel("CPR (%)")
	ax.set_ylabel("Number of images")
	ax.legend()
	plt.tight_layout()
	out = os.path.join(GRAPHS_DIR, "01_cpr_histogram.png")
	plt.savefig(out, dpi=150, bbox_inches="tight")
	plt.close()
	print("  [OK] CPR histogram saved")


def _plot_damage_bars(df):
	counts = df["damage_level"].value_counts().reindex(["low", "moderate", "high"], fill_value=0)
	colors = {"low": "#43A047", "moderate": "#FBC02D", "high": "#E53935"}

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

	ax.set_title("Descriptive structural damage classification", fontsize=14, fontweight="bold")
	ax.set_xlabel("Descriptive severity")
	ax.set_ylabel("Number of images")
	plt.tight_layout()
	out = os.path.join(GRAPHS_DIR, "02_severity_categories.png")
	plt.savefig(out, dpi=150, bbox_inches="tight")
	plt.close()
	print("  [OK] severity chart saved")


def _plot_mosaic(df, best_operator="canny", n=9):
	selected = df.head(n)["image"].tolist()
	if not selected:
		print("  [!] no images available to build the mosaic")
		return

	rows = len(selected)
	fig, axes = plt.subplots(rows, 3, figsize=(12, rows * 3.2), squeeze=False)
	fig.suptitle(
		f"Original | Mask ({best_operator}) | Overlay",
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

		row = df[df["image"] == name].iloc[0]
		label = f"{name} | CPR: {row['cpr']:.3f}% | {row['damage_level']}"

		axes[i][0].imshow(orig, cmap="gray")
		axes[i][0].set_title("Original", fontsize=8)
		axes[i][0].axis("off")

		axes[i][1].imshow(mask, cmap="gray")
		axes[i][1].set_title("Mask", fontsize=8)
		axes[i][1].axis("off")

		axes[i][2].imshow(overlay)
		axes[i][2].set_title(label, fontsize=7)
		axes[i][2].axis("off")

	plt.tight_layout()
	out = os.path.join(VIS_DIR, "03_comparison_mosaic.png")
	plt.savefig(out, dpi=130, bbox_inches="tight")
	plt.close()
	print("  [OK] comparison mosaic saved")


def _plot_fft_spectrum(df):
	if "cpr" not in df.columns or df.empty:
		return

	worst_idx = df["cpr"].idxmax()
	worst_img = df.loc[worst_idx, "image"]
	img_path = os.path.join(PROCESSED_DIR, worst_img)
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	if img is None:
		print(f"  [!] could not load image for FFT: {img_path}")
		return

	freq = np.fft.fft2(img)
	freq_shift = np.fft.fftshift(freq)
	magnitude = 20 * np.log(np.abs(freq_shift) + 1)

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	fig.suptitle(f"FFT analysis - {worst_img}", fontsize=13, fontweight="bold")

	axes[0].imshow(img, cmap="gray")
	axes[0].set_title("Preprocessed image")
	axes[0].axis("off")

	axes[1].imshow(magnitude, cmap="inferno")
	axes[1].set_title("Magnitude spectrum")
	axes[1].axis("off")

	plt.tight_layout()
	out = os.path.join(GRAPHS_DIR, "04_fft_spectrum.png")
	plt.savefig(out, dpi=150, bbox_inches="tight")
	plt.close()
	print("  [OK] FFT spectrum saved")


def run(best_operator="canny"):
	"""Generate all final visualizations."""
	os.makedirs(GRAPHS_DIR, exist_ok=True)
	os.makedirs(VIS_DIR, exist_ok=True)

	if not os.path.exists(SUMMARY_CSV):
		print(f"  error: {SUMMARY_CSV} not found; run 04_comparison.py first")
		return

	df = pd.read_csv(SUMMARY_CSV)
	if df.empty:
		print(f"  error: {SUMMARY_CSV} is empty; there is no data to visualize")
		return

	required = {"image", "cpr", "damage_level"}
	missing = [c for c in required if c not in df.columns]
	if missing:
		print(f"  error: required columns are missing from {SUMMARY_CSV}: {missing}")
		return

	print(f"  generating visualizations for {len(df)} images...")
	try:
		_plot_cpr_histogram(df)
		_plot_damage_bars(df)
		_plot_mosaic(df, best_operator=best_operator)
		_plot_fft_spectrum(df)
		print("  [OK] visualizations generated successfully")
	except Exception as e:
		print(f"  [ERR] visualization generation failed: {e}")


if __name__ == "__main__":
	run()
