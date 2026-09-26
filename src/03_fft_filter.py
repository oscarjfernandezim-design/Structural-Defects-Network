"""Compare six edge detectors using CPR and a connected-component score."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MASKS_DIR = "results/masks"
GRAPHS_DIR = "results/graphs"
OPERATORS = ("canny", "laplacian", "sobel", "prewitt", "roberts", "fft")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
OPERATOR_LABELS = {"laplacian": "Laplacian", "fft": "FFT"}


def calculate_cpr(mask):
    """Calculate the percentage of active pixels."""
    return np.count_nonzero(mask) / mask.size * 100


def calculate_uniformity(mask):
    """Estimate detection localization using connected-component entropy.

    Higher values indicate more localized detections; lower values indicate
    more spatially distributed noise.
    """
    component_count, _, statistics, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if component_count <= 1:
        return 0.0

    areas = statistics[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return 0.0

    normalized_areas = areas / np.sum(areas)
    entropy = -np.sum(
        normalized_areas[normalized_areas > 0]
        * np.log2(normalized_areas[normalized_areas > 0] + 1e-10)
    )
    maximum_entropy = np.log2(len(areas))
    uniformity = (
        (maximum_entropy - entropy) / maximum_entropy
        if maximum_entropy > 0
        else 0
    )
    return max(0, min(1, uniformity))


def run():
    """Compare operators and generate charts."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    measurements = {
        operator: {"cpr": [], "uniformity": []} for operator in OPERATORS
    }

    first_operator_dir = os.path.join(MASKS_DIR, OPERATORS[0])
    if not os.path.exists(first_operator_dir):
        print("  error: no masks found; run 02_edge_detection.py first")
        return

    image_names = sorted(
        name
        for name in os.listdir(first_operator_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_names:
        print("  error: no images found in the mask directory")
        return

    print(
        f"  comparing {len(OPERATORS)} operators across "
        f"{len(image_names)} images..."
    )
    for image_name in image_names:
        for operator in OPERATORS:
            mask_path = os.path.join(MASKS_DIR, operator, image_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            measurements[operator]["cpr"].append(calculate_cpr(mask))
            measurements[operator]["uniformity"].append(
                calculate_uniformity(mask)
            )

    rows = []
    for operator in OPERATORS:
        cpr_values = measurements[operator]["cpr"]
        uniformity_values = measurements[operator]["uniformity"]
        rows.append(
            {
                "operator": OPERATOR_LABELS.get(operator, operator.capitalize()),
                "mean_cpr": round(float(np.mean(cpr_values)), 4)
                if cpr_values
                else 0,
                "mean_uniformity": round(float(np.mean(uniformity_values)), 4)
                if uniformity_values
                else 0,
            }
        )

    results = pd.DataFrame(rows)
    print("\n  operator comparison:")
    print(results.to_string(index=False))
    results.to_csv(
        os.path.join(GRAPHS_DIR, "operator_comparison.csv"), index=False
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle("Edge detector comparison", fontsize=15, fontweight="bold")
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    axes[0].bar(
        results["operator"],
        results["mean_cpr"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[0].set_title(
        "Mean CPR (%)\n(lower is not necessarily more accurate)",
        fontweight="bold",
    )
    axes[0].set_ylabel("CPR (%)")
    axes[0].set_xlabel("Operator")
    axes[0].grid(axis="y", alpha=0.3)
    for index, value in enumerate(results["mean_cpr"]):
        axes[0].text(
            index,
            value + 0.15,
            f"{value:.3f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    axes[1].bar(
        results["operator"],
        results["mean_uniformity"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1].set_title(
        "Mean component uniformity\n(higher indicates more localization)",
        fontweight="bold",
    )
    axes[1].set_ylabel("Uniformity (0-1)")
    axes[1].set_xlabel("Operator")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)
    for index, value in enumerate(results["mean_uniformity"]):
        axes[1].text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    chart_path = os.path.join(GRAPHS_DIR, "operator_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] chart saved to {chart_path}")

    normalized_cpr = 1 - results["mean_cpr"] / results["mean_cpr"].max()
    results["score"] = (normalized_cpr + results["mean_uniformity"]) / 2
    best_index = results["score"].idxmax()
    best_operator = results.loc[best_index, "operator"].lower()

    print(
        f"\n  top-ranked operator "
        f"(score={results.loc[best_index, 'score']:.4f}): "
        f"{best_operator.upper()}"
    )
    print(f"    - CPR: {results.loc[best_index, 'mean_cpr']:.4f}%")
    print(f"    - Uniformity: {results.loc[best_index, 'mean_uniformity']:.4f}")
    return best_operator


if __name__ == "__main__":
    run()
