# Structural Defects Network

A reproducible computer-vision baseline for exploring crack candidates in
infrastructure images. This repository contains classical image-processing
methods; **it is not a certified structural inspection system**.

## What it does

1. Converts images to grayscale, resizes them to `256x256`, and applies a
   median filter.
2. Generates candidate masks with Canny, Laplacian, Sobel, Prewitt, Roberts,
   and a high-pass FFT filter.
3. Calculates crack pixel ratio (CPR), connected components, and descriptive
   severity.
4. When labels are available in `data/annotated/Cracked` and
   `data/annotated/No-Cracked`, calibrates a CPR rule on a training split and
   reports classification results on a separate test split.

CPR measures active edge/texture pixels; it is not a physical measurement of
a crack. Pixel-level segmentation evaluation requires human-annotated masks.

## Installation

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project structure

```text
.
├── run_pipeline.py
├── app.py
├── requirements.txt
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_edge_detection.py
│   ├── 03_fft_filter.py
│   ├── 04_comparison.py
│   ├── 06_visualization.py
│   └── 07_evaluate.py
├── data/
│   ├── raw/                  # input images
│   ├── processed/            # generated
│   └── annotated/
│       ├── Cracked/          # positive image labels
│       └── No-Cracked/       # negative image labels
└── results/                  # generated masks, charts, and reports
```

Datasets and generated artifacts are excluded from Git via `.gitignore`.

## Run the pipeline

Place images in `data/raw/` and run:

```powershell
python run_pipeline.py
```

The pipeline exits with a non-zero status if a stage cannot produce output.
The project root is resolved automatically, so the script can be launched
from another directory:

```powershell
python C:\path\to\Structural-Defects-Network\run_pipeline.py
```

## Interactive demo

Launch the Streamlit application and open the local URL printed by Streamlit:

```powershell
streamlit run app.py
```

Upload an image to view the original, the Prewitt candidate mask, CPR, and a
heuristic image-level prediction. The demo is experimental and must not be
used for safety-critical decisions.

## Evaluate labeled images

Run:

```powershell
python src\07_evaluate.py --folder-labels
```

The evaluation report is written to:

- `results/evaluation/classification_by_image.csv`
- `results/evaluation/classification_summary.csv`
- `results/evaluation/classification_report.md`

Only images present in both the labeled folders and generated mask folders
are scored. Missing predictions are explicitly reported.

## Evaluation protocol

For each class, 70% of filenames in sorted order are used to select the CPR
direction (`CPR >= threshold` or `CPR <= threshold`) and threshold by F1. The
remaining 30% are held out for testing. This is a small, deterministic
baseline split, not cross-validation; results may vary substantially with
another dataset.

Folder labels evaluate image classification. `IoU` and `Dice` are meaningful
only when human pixel masks are provided in `data/annotated/images` and
`data/annotated/masks`.

## Limitations

- Classical operators detect edges, texture, and noise; they do not
  semantically identify cracks.
- Fixed-size resizing may discard physical scale and very thin cracks.
- The current image labels do not include pixel-level masks.
- The small dataset gives limited statistical confidence.
- Expert inspection and independent validation are required before any
  structural safety decision.

## Tests

The test suite can be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
