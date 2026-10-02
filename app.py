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

st.markdown(
    """
    <style>
    :root {
        --ink: #14252d;
        --muted: #62737b;
        --line: #e2e9e8;
        --teal: #087f78;
        --teal-dark: #075b58;
        --paper: #f4f7f6;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(8,127,120,.07), transparent 34rem),
            var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1160px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }
    [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    .topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.15rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: .65rem;
        color: var(--ink);
        font-size: .88rem;
        font-weight: 750;
        letter-spacing: .02em;
    }
    .brand-mark {
        display: grid;
        width: 2rem;
        height: 2rem;
        place-items: center;
        border-radius: .65rem;
        background: #d8f0ec;
        color: var(--teal-dark);
        font-size: 1.1rem;
    }
    .top-badge, .step-pill {
        border: 1px solid #d7e4e1;
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        color: #48635f;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .08em;
        padding: .42rem .72rem;
        text-transform: uppercase;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid #173e40;
        border-radius: 1.25rem;
        background:
            radial-gradient(circle at 84% 20%, rgba(76,206,186,.2), transparent 17rem),
            linear-gradient(118deg, #10272d 0%, #153b3d 57%, #12665f 100%);
        box-shadow: 0 18px 42px rgba(19,54,54,.14);
        color: white;
        padding: clamp(1.5rem, 4vw, 3.1rem);
    }
    .hero:after {
        position: absolute;
        right: -2rem;
        bottom: -8rem;
        width: 22rem;
        height: 22rem;
        border: 1px solid rgba(255,255,255,.11);
        border-radius: 50%;
        box-shadow: 0 0 0 2.5rem rgba(255,255,255,.025), 0 0 0 5rem rgba(255,255,255,.02);
        content: "";
        pointer-events: none;
    }
    .hero-copy { position: relative; z-index: 1; max-width: 690px; }
    .hero-kicker {
        color: #83dfd0;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .65rem 0 .75rem;
        color: #fff;
        font-size: clamp(2.1rem, 5vw, 3.5rem);
        font-weight: 760;
        letter-spacing: -.045em;
        line-height: 1.04;
    }
    .hero p {
        max-width: 590px;
        margin: 0;
        color: #d1e2df;
        font-size: 1.02rem;
        line-height: 1.65;
    }
    .steps {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.55rem;
    }
    .step-pill {
        border-color: rgba(255,255,255,.19);
        background: rgba(255,255,255,.08);
        color: #e0f1ee;
        font-size: .67rem;
        letter-spacing: .045em;
    }
    .section-heading {
        margin: 2rem 0 .75rem;
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 750;
        letter-spacing: -.02em;
    }
    .section-subtitle {
        margin-top: -.4rem;
        margin-bottom: 1rem;
        color: var(--muted);
        font-size: .9rem;
    }
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #91bdb6;
        border-radius: 1rem;
        background: rgba(255,255,255,.76);
        padding: .8rem;
        transition: border-color .18s ease, background .18s ease;
    }
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
        color: #42575d;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #f8fcfb;
        color: #42575d;
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #536970;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--teal);
        background: #fff;
    }
    [data-testid="stFileUploader"] section {
        border: 0;
        background: transparent;
    }
    [data-testid="stFileUploader"] button {
        border: 0;
        border-radius: .65rem;
        background: var(--teal);
        color: white;
        font-weight: 700;
    }
    [data-testid="stFileUploader"] button:hover {
        background: var(--teal-dark);
        color: white;
    }
    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 2rem 0 .85rem;
    }
    .result-title {
        color: var(--ink);
        font-size: 1.2rem;
        font-weight: 750;
        letter-spacing: -.02em;
    }
    .result-tag {
        border: 1px solid #d8e8e4;
        border-radius: 999px;
        background: #e9f5f2;
        color: #176b62;
        font-size: .7rem;
        font-weight: 750;
        letter-spacing: .07em;
        padding: .4rem .68rem;
        text-transform: uppercase;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 1rem;
        background: rgba(255,255,255,.88);
        box-shadow: 0 8px 26px rgba(20,48,50,.045);
    }
    .image-label {
        margin-bottom: .65rem;
        color: #42575d;
        font-size: .82rem;
        font-weight: 700;
    }
    .metric-card {
        min-height: 132px;
        border: 1px solid var(--line);
        border-radius: .95rem;
        background: #fff;
        padding: 1.05rem 1.1rem;
    }
    .metric-label {
        color: var(--muted);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .045em;
        text-transform: uppercase;
    }
    .metric-value {
        margin-top: .6rem;
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 760;
        letter-spacing: -.035em;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .metric-note { margin-top: .45rem; color: #819096; font-size: .76rem; }
    .metric-positive { color: #a04c27; }
    .metric-negative { color: #08756c; }
    .notice {
        border: 1px solid #f0dfac;
        border-radius: .85rem;
        background: #fff9e9;
        color: #715a25;
        font-size: .83rem;
        line-height: 1.55;
        margin-top: 1rem;
        padding: .85rem 1rem;
    }
    .empty-state {
        border: 1px solid var(--line);
        border-radius: 1rem;
        background: rgba(255,255,255,.7);
        color: var(--muted);
        margin-top: 1.3rem;
        padding: 1.1rem 1.25rem;
    }
    .empty-state strong { color: var(--ink); }
    .footer {
        border-top: 1px solid #e0e8e6;
        color: #819096;
        font-size: .76rem;
        line-height: 1.6;
        margin-top: 2.3rem;
        padding-top: 1rem;
    }
    @media (max-width: 640px) {
        .block-container { padding-top: 1rem; }
        .topline { margin-bottom: .8rem; }
        .hero { border-radius: .95rem; }
        .metric-card { min-height: 112px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topline">
      <div class="brand"><span class="brand-mark">⌁</span> FIELDNOTE / VISION</div>
      <span class="top-badge">Experimental baseline</span>
    </div>
    <section class="hero">
      <div class="hero-copy">
        <div class="hero-kicker">Infrastructure image analysis</div>
        <h1>See the details.<br>Surface possible defects.</h1>
        <p>Explore an edge-based crack candidate map from a single image.
        A clear visual aid for experimentation — not a structural diagnosis.</p>
        <div class="steps">
          <span class="step-pill">01&nbsp; Upload image</span>
          <span class="step-pill">02&nbsp; Map edges</span>
          <span class="step-pill">03&nbsp; Review indicators</span>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Start an inspection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Choose a clear JPG or PNG image of a surface or structure.</div>',
    unsafe_allow_html=True,
)

uploaded_image = st.file_uploader(
    "Upload an infrastructure image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG and PNG.",
)

if uploaded_image is not None:
    encoded = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("The uploaded file could not be decoded as an image.")
        st.stop()

    original_height, original_width = image_bgr.shape[:2]
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    preprocessed = cv2.medianBlur(resized, 3)
    mask = detectors.clean_mask(detectors.EDGE_DETECTORS[OPERATOR](preprocessed))

    cpr = float(np.count_nonzero(mask) / mask.size * 100)
    cracked = cpr <= CPR_THRESHOLD
    label = "Possible crack" if cracked else "No crack detected"
    metric_class = "metric-positive" if cracked else "metric-negative"

    st.markdown(
        """
        <div class="result-header">
          <div class="result-title">Analysis overview</div>
          <span class="result-tag">Prewitt edge map</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"File: {uploaded_image.name} · Original resolution: {original_width} × {original_height}px")

    original_column, mask_column = st.columns(2, gap="large")
    with original_column:
        with st.container(border=True):
            st.markdown('<div class="image-label">01 / Uploaded image</div>', unsafe_allow_html=True)
            st.image(original_rgb, use_container_width=True)
    with mask_column:
        with st.container(border=True):
            st.markdown('<div class="image-label">02 / Candidate edge mask</div>', unsafe_allow_html=True)
            st.image(mask, clamp=True, use_container_width=True)

    prediction_column, cpr_column, threshold_column = st.columns(3, gap="medium")
    with prediction_column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Heuristic signal</div>'
            f'<div class="metric-value {metric_class}">{label}</div>'
            '<div class="metric-note">Image-level baseline estimate</div></div>',
            unsafe_allow_html=True,
        )
    with cpr_column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Candidate pixel ratio</div>'
            f'<div class="metric-value">{cpr:.3f}%</div>'
            '<div class="metric-note">Active pixels in generated mask</div></div>',
            unsafe_allow_html=True,
        )
    with threshold_column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Reference threshold</div>'
            f'<div class="metric-value">≤ {CPR_THRESHOLD:.3f}%</div>'
            '<div class="metric-note">Current baseline calibration</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="notice"><strong>Interpret with care.</strong> This classic vision '
        'baseline responds to edges, texture, and noise. Its candidate pixel ratio '
        'is not crack severity, and its heuristic signal does not confirm damage. '
        'Do not use it for safety or engineering decisions.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="empty-state"><strong>Your analysis will appear here.</strong> '
        'Upload an image above to compare the source with its candidate mask and '
        'review the baseline indicators.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">STRUCTURAL DEFECTS NETWORK &nbsp;·&nbsp; '
    'Computer-vision research demo &nbsp;·&nbsp; Not for safety-critical use</div>',
    unsafe_allow_html=True,
)
