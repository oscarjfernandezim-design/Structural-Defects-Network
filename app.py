"""Streamlit demo for interactive crack-candidate analysis."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parent
DETECTOR_PATH = ROOT / "src" / "02_edge_detection.py"
IMAGE_SIZE = (256, 256)
OPERATOR = "prewitt"
CPR_THRESHOLD = 13.092


def _load_detectors():
    spec = importlib.util.spec_from_file_location("edge_detection", DETECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the edge detector module: {DETECTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detectors = _load_detectors()

st.set_page_config(
    page_title="Structural Defects Network",
    page_icon="🧱",
    layout="wide",
)
st.title("Structural Defects Network")
st.write("Upload an image to view a candidate mask and image-level classification.")
st.warning(
    "Experimental demo: this baseline detects edges and texture; it does not "
    "confirm structural damage. Do not use predictions for safety decisions."
)

st.caption("Baseline detector: Prewitt")
uploaded_image = st.file_uploader(
    "Upload an infrastructure image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    encoded = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("The uploaded file could not be decoded as an image.")
        st.stop()

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    preprocessed = cv2.medianBlur(resized, 3)
    mask = detectors.clean_mask(
        detectors.EDGE_DETECTORS[OPERATOR](preprocessed)
    )

    cpr = float(np.count_nonzero(mask) / mask.size * 100)
    cracked = cpr <= CPR_THRESHOLD

    original_column, mask_column = st.columns(2)
    with original_column:
        st.subheader("Uploaded image")
        st.image(original_rgb, use_container_width=True)
    with mask_column:
        st.subheader("Candidate mask — Prewitt")
        st.image(mask, clamp=True, use_container_width=True)

    prediction_column, cpr_column, threshold_column = st.columns(3)
    with prediction_column:
        label = "Possible crack" if cracked else "No crack detected"
        st.metric("Heuristic prediction", label)
    with cpr_column:
        st.metric("CPR", f"{cpr:.3f}%")
    with threshold_column:
        st.metric("CPR threshold", f"≤ {CPR_THRESHOLD:.4f}%")

    st.caption(
        "The rule and threshold come from the current baseline calibration. "
        "CPR represents active mask pixels; it is not a severity measure or "
        "a diagnosis."
    )
