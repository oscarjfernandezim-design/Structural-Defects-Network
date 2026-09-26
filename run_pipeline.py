"""Run the complete structural damage analysis pipeline.

Usage: python run_pipeline.py
"""

import importlib
import importlib.util
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Current modules use paths relative to the project root.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(SRC_DIR))


def load_module(name, path):
    """Dynamically load a module from a file."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_dependencies():
    """Check dependencies before starting the pipeline."""
    packages = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
    }
    problems = []
    for module_name, package_name in packages.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            problems.append(f"{package_name} (not installed)")
        except ImportError as error:
            problems.append(f"{package_name} (incompatible installation: {error})")

    if problems:
        print("  [ERR] dependency issues found:")
        for problem in problems:
            print(f"        - {problem}")
        print("  [INFO] install dependencies with:")
        print("         python -m pip install -r requirements.txt")
        return False
    return True


def run_stage(name, module, function, *args, **kwargs):
    """Run a pipeline stage and report failures."""
    try:
        result = getattr(module, function)(*args, **kwargs)
    except Exception as error:
        print(f"  [ERR] {name} failed: {error}")
        return None, False
    return result, True


def main():
    if not validate_dependencies():
        return 1

    total_steps = 6

    print("=" * 70)
    print("  STRUCTURAL DAMAGE ANALYSIS PIPELINE".center(70))
    print("  Automated crack and infrastructure damage analysis".center(70))
    print("=" * 70)

    # Step 1: preprocessing.
    print(f"\n[1/{total_steps}] preprocessing images...")
    try:
        prep = load_module("prep", SRC_DIR / "01_preprocessing.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] failed to load preprocessing module: {error}")
        return 1
    result, succeeded = run_stage("preprocessing", prep, "run")
    if not succeeded or not result:
        print("  [ERR] preprocessing produced no output")
        return 1

    # Step 2: edge detection.
    print(f"\n[2/{total_steps}] detecting edges with 6 operators...")
    try:
        edge_detection = load_module("edge_detection", SRC_DIR / "02_edge_detection.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] failed to load edge detection module: {error}")
        return 1
    _, succeeded = run_stage("edge detection", edge_detection, "run")
    if not succeeded:
        return 1

    # Step 3: operator comparison.
    print(f"\n[3/{total_steps}] comparing operators...")
    try:
        comparison = load_module("comparison", SRC_DIR / "03_fft_filter.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] failed to load operator comparison module: {error}")
        return 1
    best_operator, succeeded = run_stage("operator comparison", comparison, "run")
    if not succeeded:
        return 1
    if not best_operator:
        best_operator = "canny"
        print(f"  [!] using fallback operator: {best_operator}")

    # When labels are available, select by calibrated evaluation rather than
    # the visual CPR/uniformity score.
    annotation_dir = PROJECT_ROOT / "data" / "annotated"
    if (annotation_dir / "Cracked").is_dir() and (annotation_dir / "No-Cracked").is_dir():
        try:
            evaluator = load_module("evaluator", SRC_DIR / "07_evaluate.py")
            _, summary = evaluator.evaluate_folder_labels(
                annotation_dir=annotation_dir,
                prediction_dir=PROJECT_ROOT / "results" / "masks",
            )
            if not summary.empty and summary["f1"].notna().any():
                best_operator = str(summary.iloc[0]["operator"])
                print(
                    f"  [OK] operator selected by labeled test F1: "
                    f"{best_operator.upper()}"
                )
        except (ImportError, OSError, ValueError) as error:
            print(f"  [!] labeled evaluation could not be used: {error}")

    # Step 4: metric calculation.
    print(f"\n[4/{total_steps}] calculating metrics with {best_operator.upper()}...")
    try:
        metrics = load_module("metrics", SRC_DIR / "04_comparison.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] failed to load metrics module: {error}")
        return 1
    metrics_df, succeeded = run_stage(
        "metrics", metrics, "run", best_operator=best_operator
    )
    if not succeeded or metrics_df is None:
        print("  [ERR] metrics could not be calculated")
        return 1

    # Step 5: visualization.
    print(f"\n[5/{total_steps}] generating visualizations...")
    try:
        visualization = load_module("visualization", SRC_DIR / "06_visualization.py")
    except (ImportError, OSError) as error:
        print(f"  [ERR] failed to load visualization module: {error}")
        return 1
    _, succeeded = run_stage(
        "visualization", visualization, "run", best_operator=best_operator
    )
    if not succeeded:
        return 1

    # Step 6: final summary.
    print(f"\n[6/{total_steps}] generating final report...")
    print("\n" + "=" * 70)
    print("  [OK] ANALYSIS COMPLETE".center(70))
    print("=" * 70)
    print(f"   Key results:")
    print(f"     - Selected operator: {best_operator.upper()}")
    print(f"     - Images processed: {len(metrics_df)}")
    print(f"     - Mean CPR: {metrics_df['cpr'].mean():.3f}%")
    print(f"     - CPR range: {metrics_df['cpr'].min():.3f}% - {metrics_df['cpr'].max():.3f}%")
    print(f"\n   Generated files:")
    print(f"     - results_summary.csv")
    print(f"     - results/graphs/operator_comparison.png")
    print(f"     - results/visualizations/03_comparison_mosaic.png")
    print(f"     - results/graphs/04_fft_spectrum.png")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
