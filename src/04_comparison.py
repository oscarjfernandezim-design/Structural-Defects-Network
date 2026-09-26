"""Calculate descriptive metrics for masks from the selected operator."""

import os

import cv2
import numpy as np
import pandas as pd


MASKS_DIR = "results/masks"
SUMMARY_CSV = "results_summary.csv"
LOW_SEVERITY_THRESHOLD = 5.0
MODERATE_SEVERITY_THRESHOLD = 20.0
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def approximate_mask_length(mask):
    """Approximate mask length by counting pixels after one dilation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return np.sum(dilated > 0)


def calculate_metrics(mask):
    """Calculate descriptive metrics for a binary mask."""
    total_pixels = mask.size
    active_pixels = np.count_nonzero(mask)
    cpr = active_pixels / total_pixels * 100

    approximate_length = approximate_mask_length(mask)
    component_count_with_background, _, statistics, _ = (
        cv2.connectedComponentsWithStats(mask, connectivity=8)
    )
    component_count = component_count_with_background - 1
    component_areas = (
        statistics[1:, cv2.CC_STAT_AREA] if component_count > 0 else [0]
    )
    largest_component_area = (
        int(np.max(component_areas)) if len(component_areas) > 0 else 0
    )

    return {
        "cpr": round(cpr, 4),
        "approx_crack_length_px": int(approximate_length),
        "connected_components": component_count,
        "largest_component_area_px": largest_component_area,
    }


def classify_severity(cpr):
    """Assign a descriptive severity category based on CPR."""
    if cpr < LOW_SEVERITY_THRESHOLD:
        return "low"
    if cpr < MODERATE_SEVERITY_THRESHOLD:
        return "moderate"
    return "high"


def run(best_operator="canny"):
    """Calculate metrics for the selected operator."""
    operator_dir = os.path.join(MASKS_DIR, best_operator)
    if not os.path.exists(operator_dir):
        print(f"  error: mask directory not found for {best_operator}")
        return None

    image_names = sorted(
        name
        for name in os.listdir(operator_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_names:
        print(f"  error: no masks found for operator {best_operator}")
        return None

    print(
        f"  calculating metrics for {len(image_names)} images "
        f"with {best_operator}"
    )
    records = []
    for image_name in image_names:
        mask_path = os.path.join(operator_dir, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        try:
            metrics = calculate_metrics(mask)
            records.append(
                {
                    "image": image_name,
                    **metrics,
                    "damage_level": classify_severity(metrics["cpr"]),
                    "operator": best_operator,
                }
            )
        except Exception as error:
            print(f"  [!] failed to process {image_name}: {error}")

    if not records:
        print("  error: metrics could not be calculated for any image")
        return None

    results = pd.DataFrame(records)
    results.to_csv(SUMMARY_CSV, index=False)

    print("\n  descriptive severity summary:")
    print(results["damage_level"].value_counts().to_string())
    print("\n  CPR statistics:")
    print(f"    - Mean: {results['cpr'].mean():.3f}%")
    print(f"    - Minimum: {results['cpr'].min():.3f}%")
    print(f"    - Maximum: {results['cpr'].max():.3f}%")
    print(f"\n  [OK] saved {len(results)} records to {SUMMARY_CSV}")
    return results


if __name__ == "__main__":
    run()
