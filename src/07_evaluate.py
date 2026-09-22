"""Evalua las mascaras del baseline contra mascaras anotadas.

Estructura esperada:
    data/annotated/images/<nombre>.(jpg|jpeg|png)
    data/annotated/masks/<mismo_nombre>.(png|jpg|jpeg)

Las mascaras anotadas se consideran binarias: todo pixel distinto de cero es
positivo. No se generan anotaciones automaticamente.
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
OPERATORS = ("canny", "laplaciano", "sobel", "prewitt", "roberts", "fft")
LABEL_DIRS = {"Cracked": 1, "No-Cracked": 0}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _binary_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"no se pudo leer la mascara: {path}")
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def calculate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calcula metricas binarias por imagen, incluyendo casos vacios."""
    if prediction.shape != target.shape:
        raise ValueError("prediction y target deben tener la misma forma")

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
    """Evalua un operador usando nombres de archivo compartidos."""
    images = _indexed_files(image_dir)
    targets = _indexed_files(target_dir)
    predictions = _indexed_files(prediction_dir)
    names = sorted(set(images) & set(targets) & set(predictions))
    if not names:
        raise ValueError(
            f"no hay nombres coincidentes entre imagenes, mascaras y predicciones "
            f"para {operator}"
        )

    rows = []
    for name in names:
        target = _binary_mask(targets[name])
        prediction = _binary_mask(predictions[name], target.shape)
        metrics = calculate_metrics(prediction, target)
        rows.append({"imagen": targets[name].name, "operador": operator, **metrics})
    return pd.DataFrame(rows)


def evaluate_dataset(
    image_dir: Path = ANNOTATED_DIR / "images",
    target_dir: Path = ANNOTATED_DIR / "masks",
    prediction_dir: Path = PREDICTIONS_DIR,
    operators: tuple[str, ...] = OPERATORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalua todos los operadores y devuelve detalle y resumen."""
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"no existe el directorio de imagenes anotadas: {image_dir}"
        )
    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"no existe el directorio de mascaras anotadas: {target_dir}"
        )

    details = []
    for operator in operators:
        operator_dir = prediction_dir / operator
        if not operator_dir.is_dir():
            raise FileNotFoundError(
                f"no existe el directorio de predicciones para {operator}: {operator_dir}"
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
        detail_df.groupby("operador", as_index=False)[metric_columns]
        .mean()
        .sort_values("iou", ascending=False)
    )
    return detail_df, summary_df


def _load_folder_labels(annotation_dir: Path) -> dict[str, tuple[int, str]]:
    labels = {}
    for folder_name, label in LABEL_DIRS.items():
        folder = annotation_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"no existe la carpeta etiquetada: {folder}")
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                stem = path.stem
                if stem in labels:
                    raise ValueError(f"la imagen aparece en mas de una etiqueta: {stem}")
                labels[stem] = (label, folder_name)
    if not labels:
        raise ValueError(f"no hay imagenes etiquetadas en {annotation_dir}")
    return labels


def evaluate_folder_labels(
    annotation_dir: Path = ANNOTATED_DIR,
    prediction_dir: Path = PREDICTIONS_DIR,
    operators: tuple[str, ...] = OPERATORS,
    cpr_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibra CPR en train y evalua en test, sin contaminar la validacion."""
    labels = _load_folder_labels(annotation_dir)
    base_rows = []
    for operator in operators:
        operator_dir = prediction_dir / operator
        if not operator_dir.is_dir():
            raise FileNotFoundError(
                f"no existe el directorio de predicciones para {operator}: {operator_dir}"
            )
        predictions = _indexed_files(operator_dir)
        for stem, (actual, actual_label) in labels.items():
            prediction_path = predictions.get(stem)
            if prediction_path is None:
                base_rows.append(
                    {
                        "imagen": f"{stem} (sin prediccion)",
                        "operador": operator,
                        "etiqueta_real": actual_label,
                        "real_grieta": actual,
                        "cpr": np.nan,
                        "umbral_cpr": cpr_threshold,
                        "prediccion": "SIN_PREDICCION",
                        "predice_grieta": np.nan,
                        "correcto": False,
                        "resultado": "MISSING",
                        "particion": "missing",
                    }
                )
                continue
            mask = _binary_mask(prediction_path)
            cpr = np.count_nonzero(mask) / mask.size * 100
            base_rows.append(
                {
                    "imagen": prediction_path.name,
                    "operador": operator,
                    "etiqueta_real": actual_label,
                    "real_grieta": actual,
                    "cpr": round(cpr, 4),
                    "umbral_cpr": np.nan,
                    "prediccion": "NO_CALCULADA",
                    "predice_grieta": np.nan,
                    "correcto": np.nan,
                    "resultado": "UNSET",
                    "particion": "unset",
                }
            )

    detail_df = pd.DataFrame(base_rows)
    valid = detail_df[detail_df["resultado"] != "MISSING"].copy()
    valid["particion"] = "test"
    for operator in valid["operador"].unique():
        operator_rows = valid[valid["operador"] == operator]
        for label in (0, 1):
            indexes = operator_rows.index[operator_rows["real_grieta"] == label].tolist()
            split_at = max(1, int(len(indexes) * 0.7))
            valid.loc[indexes[:split_at], "particion"] = "train"
    detail_df.loc[valid.index, "particion"] = valid["particion"]

    def score_rule(y_true: np.ndarray, values: np.ndarray, threshold: float, reverse: bool) -> float:
        predicted = values <= threshold if reverse else values >= threshold
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))
        fn = np.sum((predicted == 0) & (y_true == 1))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary_rows = []
    for operator, group in detail_df.groupby("operador", sort=False):
        train = group[group["particion"] == "train"]
        test = group[group["particion"] == "test"]
        values = train["cpr"].to_numpy(dtype=float)
        labels_train = train["real_grieta"].to_numpy(dtype=int)
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
        detail_df.loc[evaluated.index, "umbral_cpr"] = fitted_threshold
        detail_df.loc[evaluated.index, "predice_grieta"] = predicted
        detail_df.loc[evaluated.index, "prediccion"] = np.where(
            predicted, "Cracked", "No-Cracked"
        )
        actual_test = evaluated["real_grieta"].to_numpy(dtype=int)
        detail_df.loc[evaluated.index, "correcto"] = predicted == actual_test
        detail_df.loc[evaluated.index, "resultado"] = [
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
                "operador": operator,
                "imagenes": len(evaluated),
                "imagenes_etiquetadas": len(group),
                "sin_prediccion": int((group["resultado"] == "MISSING").sum()),
                "correctas": int(np.sum(predicted == actual)),
                "accuracy": float(np.mean(predicted == actual)),
                "particion": "test",
                "regla": "<=" if reverse else ">=",
                "umbral_calibrado": fitted_threshold,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "cpr_promedio": evaluated["cpr"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    return detail_df, summary_df


def write_folder_report(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Escribe un informe legible con resultados y errores por imagen."""
    lines = [
        "# Evaluacion de clasificacion de grietas",
        "",
        "Comparacion de las carpetas `Cracked` y `No-Cracked` contra las "
        "mascaras generadas por cada operador.",
        "",
        "El umbral y la direccion de la regla CPR se calibran solo con el 70% "
        "de las imagenes disponibles por clase. Las metricas reportadas usan "
        "el 30% restante, que no participa en el ajuste.",
        "",
        "## Resumen por operador",
        "",
        "```text",
        summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Resultados por imagen",
        "",
        "```text",
        detail_df.to_string(index=False),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evalua mascaras del baseline contra anotaciones binarias."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ANNOTATED_DIR / "images",
        help="directorio con imagenes anotadas",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=ANNOTATED_DIR / "masks",
        help="directorio con mascaras anotadas",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="directorio results/masks del pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directorio para CSV de evaluacion",
    )
    parser.add_argument(
        "--cpr-threshold",
        type=float,
        default=5.0,
        help="CPR minimo para predecir Cracked (por defecto: 5.0)",
    )
    parser.add_argument(
        "--folder-labels",
        action="store_true",
        help="evalua carpetas Cracked y No-Cracked en lugar de mascaras anotadas",
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
        print("[INFO] agrega imagenes y mascaras anotadas antes de evaluar.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.folder_labels:
        detail_path = args.output_dir / "classification_by_image.csv"
        summary_path = args.output_dir / "classification_summary.csv"
        report_path = args.output_dir / "classification_report.md"
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        write_folder_report(detail_df, summary_df, report_path)
        print(f"[OK] detalle por imagen guardado en {detail_path}")
        print(f"[OK] resumen guardado en {summary_path}")
        print(f"[OK] informe guardado en {report_path}")
        print(summary_df.to_string(index=False))
        return 0

    detail_path = args.output_dir / "baseline_metrics_by_image.csv"
    summary_path = args.output_dir / "baseline_metrics_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] {len(detail_df)} evaluaciones guardadas en {detail_path}")
    print(f"[OK] resumen guardado en {summary_path}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
