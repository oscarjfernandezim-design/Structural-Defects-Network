"""Evaluate baseline masks against annotated masks.

Expected structure:
    data/annotated/images/<name>.(jpg|jpeg|png)
    data/annotated/masks/<same_name>.(png|jpg|jpeg)

Annotated masks are treated as binary: any non-zero pixel is positive.
Annotations are not generated automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "masks"
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
OPERATORS = ("canny", "laplacian", "sobel", "prewitt", "roberts", "fft")
LABEL_DIRS = {"Cracked": 1, "No-Cracked": 0}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _binary_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def calculate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calculate binary per-image metrics, including empty-mask cases."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = np.count_nonzero(prediction & target)
    false_positive = np.count_nonzero(prediction & ~target)
    true_negative = np.count_nonzero(~prediction & ~target)

    predicted_area = np.count_nonzero(prediction)
    target_area = np.count_nonzero(target)
    intersection = true_positive
    union = np.count_nonzero(prediction | target)

    precision = true_positive / (true_positive + false_positive) if predicted_area else (
        1.0 if target_area == 0 else 0.0
    )
    recall = true_positive / target_area if target_area else (
        1.0 if predicted_area == 0 else 0.0
    )
    iou = intersection / union if union else 1.0
    dice = (
        2 * intersection / (predicted_area + target_area)
        if predicted_area + target_area
        else 1.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 1.0
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "target_cpr": target_area / target.size * 100,
        "predicted_cpr": predicted_area / prediction.size * 100,
    }


def _indexed_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def evaluate_operator(
    operator: str,
    image_dir: Path,
    target_dir: Path,
    prediction_dir: Path,
) -> pd.DataFrame:
    """Evaluate one operator using matching filenames."""
    images = _indexed_files(image_dir)
    targets = _indexed_files(target_dir)
    predictions = _indexed_files(prediction_dir)
    names = sorted(set(images) & set(targets) & set(predictions))
    if not names:
        raise ValueError(
            f"No matching filenames among images, masks, and predictions "
            f"for {operator}"
        )

    rows = []
    for name in names:
        target = _binary_mask(targets[name])
        prediction = _binary_mask(predictions[name], target.shape)
        metrics = calculate_metrics(prediction, target)
        rows.append(
            {
                "image": targets[name].name,
                "operator": operator,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_dataset(
    image_dir: Path = ANNOTATED_DIR / "images",
    target_dir: Path = ANNOTATED_DIR / "masks",
    prediction_dir: Path = PREDICTIONS_DIR,
    operators: tuple[str, ...] = OPERATORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all operators and return per-image details and a summary."""
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Annotated image directory does not exist: {image_dir}"
        )
    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"Annotated mask directory does not exist: {target_dir}"
        )

    details = []
    for operator in operators:
        operator_dir = prediction_dir / operator
        if not operator_dir.is_dir():
            raise FileNotFoundError(
                f"Prediction directory for {operator} does not exist: {operator_dir}"
            )
        details.append(evaluate_operator(operator, image_dir, target_dir, operator_dir))

    detail_df = pd.concat(details, ignore_index=True)
    metric_columns = [
        "iou",
        "dice",
        "precision",
        "recall",
        "specificity",
        "target_cpr",
        "predicted_cpr",
    ]
    summary_df = (
        detail_df.groupby("operator", as_index=False)[metric_columns]
        .mean()
        .sort_values("iou", ascending=False)
    )
    return detail_df, summary_df


def _load_folder_labels(annotation_dir: Path) -> dict[str, tuple[int, str]]:
    labels = {}
    for folder_name, label in LABEL_DIRS.items():
        folder = annotation_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Labeled image directory does not exist: {folder}")
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                stem = path.stem
                if stem in labels:
                    raise ValueError(f"Image appears in more than one label directory: {stem}")
                labels[stem] = (label, folder_name)
    if not labels:
        raise ValueError(f"No labeled images found in {annotation_dir}")
    return labels


def evaluate_folder_labels(
    annotation_dir: Path = ANNOTATED_DIR,
    prediction_dir: Path = PREDICTIONS_DIR,
    operators: tuple[str, ...] = OPERATORS,
    cpr_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate a CPR rule on train and evaluate it on a held-out test split."""
    labels = _load_folder_labels(annotation_dir)
    base_rows = []
    for operator in operators:
        operator_dir = prediction_dir / operator
        if not operator_dir.is_dir():
            raise FileNotFoundError(
                f"Prediction directory for {operator} does not exist: {operator_dir}"
            )
        predictions = _indexed_files(operator_dir)
        for stem, (actual, actual_label) in labels.items():
            prediction_path = predictions.get(stem)
            if prediction_path is None:
                base_rows.append(
                    {
                        "image": f"{stem} (missing prediction)",
                        "operator": operator,
                        "ground_truth_label": actual_label,
                        "crack_present": actual,
                        "cpr": np.nan,
                        "cpr_threshold": cpr_threshold,
                        "prediction": "MISSING_PREDICTION",
                        "predicted_crack": np.nan,
                        "correct": False,
                        "outcome": "MISSING",
                        "split": "missing",
                    }
                )
                continue
            mask = _binary_mask(prediction_path)
            cpr = np.count_nonzero(mask) / mask.size * 100
            base_rows.append(
                {
                    "image": prediction_path.name,
                    "operator": operator,
                    "ground_truth_label": actual_label,
                    "crack_present": actual,
                    "cpr": round(cpr, 4),
                    "cpr_threshold": np.nan,
                    "prediction": "NOT_EVALUATED",
                    "predicted_crack": np.nan,
                    "correct": np.nan,
                    "outcome": "NOT_EVALUATED",
                    "split": "not_evaluated",
                }
            )

    detail_df = pd.DataFrame(base_rows)
    detail_df["correct"] = detail_df["correct"].astype(object)
    valid = detail_df[detail_df["outcome"] != "MISSING"].copy()
    valid["split"] = "test"
    for operator in valid["operator"].unique():
        operator_rows = valid[valid["operator"] == operator]
        for label in (0, 1):
            indexes = operator_rows.index[operator_rows["crack_present"] == label].tolist()
            split_at = max(1, int(len(indexes) * 0.7))
            valid.loc[indexes[:split_at], "split"] = "train"
    detail_df.loc[valid.index, "split"] = valid["split"]

    def score_rule(y_true: np.ndarray, values: np.ndarray, threshold: float, reverse: bool) -> float:
        predicted = values <= threshold if reverse else values >= threshold
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))
        fn = np.sum((predicted == 0) & (y_true == 1))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary_rows = []
    for operator, group in detail_df.groupby("operator", sort=False):
        train = group[group["split"] == "train"]
        test = group[group["split"] == "test"]
        values = train["cpr"].to_numpy(dtype=float)
        labels_train = train["crack_present"].to_numpy(dtype=int)
        candidates = np.unique(values)
        candidates = np.concatenate(([cpr_threshold], candidates))
        rules = [
            (score_rule(labels_train, values, threshold, reverse), threshold, reverse)
            for threshold in candidates
            for reverse in (False, True)
        ]
        _, fitted_threshold, reverse = max(rules, key=lambda item: item[0])
        evaluated = test.copy()
        predicted = (
            evaluated["cpr"].to_numpy() <= fitted_threshold
            if reverse
            else evaluated["cpr"].to_numpy() >= fitted_threshold
        ).astype(int)
        detail_df.loc[evaluated.index, "cpr_threshold"] = fitted_threshold
        detail_df.loc[evaluated.index, "predicted_crack"] = predicted
        detail_df.loc[evaluated.index, "prediction"] = np.where(
            predicted, "Cracked", "No-Cracked"
        )
        actual_test = evaluated["crack_present"].to_numpy(dtype=int)
        detail_df.loc[evaluated.index, "correct"] = predicted == actual_test
        detail_df.loc[evaluated.index, "outcome"] = [
            "TP" if actual and pred else
            "TN" if not actual and not pred else
            "FP" if not actual and pred else
            "FN"
            for actual, pred in zip(actual_test, predicted)
        ]
        actual = actual_test
        tp = int(np.sum((actual == 1) & (predicted == 1)))
        tn = int(np.sum((actual == 0) & (predicted == 0)))
        fp = int(np.sum((actual == 0) & (predicted == 1)))
        fn = int(np.sum((actual == 1) & (predicted == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        summary_rows.append(
            {
                "operator": operator,
                "evaluated_images": len(evaluated),
                "labeled_images": len(group),
                "missing_predictions": int((group["outcome"] == "MISSING").sum()),
                "correct_predictions": int(np.sum(predicted == actual)),
                "accuracy": float(np.mean(predicted == actual)),
                "split": "test",
                "rule": "<=" if reverse else ">=",
                "calibrated_threshold": fitted_threshold,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "mean_cpr": evaluated["cpr"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    return detail_df, summary_df


def write_folder_report(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a readable report with aggregate and per-image results."""
    lines = [
        "# Crack Classification Evaluation",
        "",
        "Comparison of `Cracked` and `No-Cracked` folder labels against "
        "masks generated by each operator.",
        "",
        "The CPR threshold and rule direction are calibrated using only 70% "
        "of available images in each class. Reported metrics use the remaining "
        "30%, which is not used during calibration.",
        "",
        "## Summary by operator",
        "",
        "```text",
        summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Per-image results",
        "",
        "```text",
        detail_df.to_string(index=False),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline masks against binary annotations."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ANNOTATED_DIR / "images",
        help="directory containing annotated images",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=ANNOTATED_DIR / "masks",
        help="directory containing annotated masks",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="pipeline results/masks directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for evaluation CSV files",
    )
    parser.add_argument(
        "--cpr-threshold",
        type=float,
        default=5.0,
        help="minimum CPR to predict Cracked (default: 5.0)",
    )
    parser.add_argument(
        "--folder-labels",
        action="store_true",
        help="evaluate Cracked and No-Cracked folders instead of annotated masks",
    )
    args = parser.parse_args()

    try:
        if args.folder_labels:
            detail_df, summary_df = evaluate_folder_labels(
                annotation_dir=ANNOTATED_DIR,
                prediction_dir=args.prediction_dir,
                cpr_threshold=args.cpr_threshold,
            )
        else:
            detail_df, summary_df = evaluate_dataset(
                image_dir=args.image_dir,
                target_dir=args.target_dir,
                prediction_dir=args.prediction_dir,
            )
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERR] {error}")
        print("[INFO] add annotated images and masks before running mask evaluation.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.folder_labels:
        detail_path = args.output_dir / "classification_by_image.csv"
        summary_path = args.output_dir / "classification_summary.csv"
        report_path = args.output_dir / "classification_report.md"
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        write_folder_report(detail_df, summary_df, report_path)
        print(f"[OK] per-image details saved to {detail_path}")
        print(f"[OK] summary saved to {summary_path}")
        print(f"[OK] report saved to {report_path}")
        print(summary_df.to_string(index=False))
        return 0

    detail_path = args.output_dir / "baseline_metrics_by_image.csv"
    summary_path = args.output_dir / "baseline_metrics_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] saved {len(detail_df)} evaluations to {detail_path}")
    print(f"[OK] summary saved to {summary_path}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
