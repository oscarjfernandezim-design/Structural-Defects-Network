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
        raise ImportError(f"No se pudo cargar el módulo de detectores: {DETECTOR_PATH}")
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
st.write("Carga una imagen para visualizar una máscara candidata y una clasificación.")
st.warning(
    "Demo experimental: este baseline detecta bordes y textura, no confirma "
    "daño estructural. No uses la predicción para decisiones de seguridad."
)

st.caption("Detector del baseline: Prewitt")
uploaded_image = st.file_uploader(
    "Sube una imagen de infraestructura",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    encoded = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("No se pudo leer el archivo como imagen.")
        st.stop()

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    preprocessed = cv2.medianBlur(resized, 3)
    mask = detectors.limpiar_ruido(
        detectors.mapa_detectores[OPERATOR](preprocessed)
    )

    cpr = float(np.count_nonzero(mask) / mask.size * 100)
    cracked = cpr <= CPR_THRESHOLD

    original_column, mask_column = st.columns(2)
    with original_column:
        st.subheader("Imagen cargada")
        st.image(original_rgb, use_container_width=True)
    with mask_column:
        st.subheader("Máscara candidata — Prewitt")
        st.image(mask, clamp=True, use_container_width=True)

    prediction_column, cpr_column, threshold_column = st.columns(3)
    with prediction_column:
        label = "Posible grieta" if cracked else "Sin grieta detectada"
        st.metric("Predicción heurística", label)
    with cpr_column:
        st.metric("CPR", f"{cpr:.3f}%")
    with threshold_column:
        st.metric("Umbral CPR", f"≤ {CPR_THRESHOLD:.4f}%")

    st.caption(
        "La regla y el umbral corresponden a la calibración del baseline "
        "actual. El CPR representa píxeles activos de la máscara; no es una "
        "medida de severidad ni un diagnóstico."
    )
