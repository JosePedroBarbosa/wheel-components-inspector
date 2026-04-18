"""
Wheel Inspector - Streamlit Demo Application.

Interactive web interface for the Wheel Components Inspector project.
Supports image upload, model selection, confidence tuning, inference
history, JSON export, and side-by-side model comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "modelos"

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wheel Inspector",
    page_icon="🔩",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px;
                border: 1px solid #3e4451; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_available_models() -> list[Path]:
    """Return sorted list of model folders inside ``modelos/`` that contain a .pt file."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        (d for d in MODELS_DIR.iterdir() if d.is_dir() and list(d.glob("*.pt"))),
        key=lambda d: d.name,
    )


def get_model_pt(folder: Path) -> Path:
    """Return the first .pt file found inside a model folder."""
    return next(iter(folder.glob("*.pt")))


@st.cache_resource
def load_model(path: str) -> YOLO:
    """Load and cache a YOLO model."""
    return YOLO(path)


def run_detection(model: YOLO, image: np.ndarray, conf: float) -> dict:
    """Run inference and return structured results."""
    results = model.predict(source=image, conf=conf, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": {
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
                "width": round(x2 - x1, 2), "height": round(y2 - y1, 2),
            },
        })

    annotated = result.plot(conf=True, labels=True, boxes=True, line_width=1, font_size=8)

    return {
        "detections": detections,
        "annotated": annotated,
        "speed_ms": result.speed["inference"],
        "resolution": f"{image.shape[1]}x{image.shape[0]}",
    }


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    """Convert a Streamlit UploadedFile to a BGR numpy array."""
    raw = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_file.seek(0)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


# ── Session state init ───────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Configuracao")

available_models = get_available_models()

if not available_models:
    st.sidebar.error("Nenhuma pasta de modelo encontrada em `modelos/`.")
    st.stop()

selected_folder = st.sidebar.selectbox("Modelo", available_models, format_func=lambda d: d.name)
conf_threshold = st.sidebar.slider("Limiar de Confianca", 0.0, 1.0, 0.25, 0.05)

enable_comparison = len(available_models) > 1 and st.sidebar.checkbox("Comparar dois modelos")
comparison_folder = None
if enable_comparison:
    other_folders = [d for d in available_models if d != selected_folder]
    if other_folders:
        comparison_folder = st.sidebar.selectbox(
            "Modelo de comparacao", other_folders, format_func=lambda d: d.name
        )

st.sidebar.divider()
st.sidebar.caption(f"Modelos disponiveis: {len(available_models)}")

# ── Main content ─────────────────────────────────────────────────────────────

st.title("Wheel Inspector")
st.markdown("Plataforma de Inspecao Visual para Deteção de Rodas, Jantes e Parafusos.")

model = load_model(str(get_model_pt(selected_folder)))

uploaded_file = st.file_uploader(
    "Arraste ou selecione a imagem para inspecao",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = decode_uploaded_image(uploaded_file)

    with st.spinner("A processar..."):
        result = run_detection(model, image, conf_threshold)

    detections = result["detections"]
    annotated = result["annotated"]

    # ── Metrics row ──────────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    m1.metric("Objetos Detetados", len(detections))
    m2.metric("Tempo Inferencia", f"{result['speed_ms']:.1f} ms")

    st.divider()

    # ── Images ───────────────────────────────────────────────────────────
    if comparison_folder:
        model_b = load_model(str(get_model_pt(comparison_folder)))
        with st.spinner("A processar modelo de comparacao..."):
            result_b = run_detection(model_b, image, conf_threshold)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### {selected_folder.name}")
            st.image(annotated, channels="BGR", width="stretch")
            st.caption(f"{len(detections)} detecoes | {result['speed_ms']:.1f} ms")
        with col_b:
            st.markdown(f"### {comparison_folder.name}")
            st.image(result_b["annotated"], channels="BGR", width="stretch")
            st.caption(
                f"{len(result_b['detections'])} detecoes | {result_b['speed_ms']:.1f} ms"
            )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Imagem Original")
            st.image(image, channels="BGR", width="stretch")
        with col2:
            st.markdown("### Bounding Boxes")
            st.image(annotated, channels="BGR", width="stretch")

    st.divider()

    # ── Tabs: table, JSON, export ────────────────────────────────────────
    tab_list, tab_json, tab_export = st.tabs(
        ["Lista de Componentes", "Output JSON", "Exportar"]
    )

    with tab_list:
        if detections:
            table_data = [
                {"Classe": d["class_name"], "Confianca": f"{d['confidence']:.2%}"}
                for d in detections
            ]
            st.table(table_data)
        else:
            st.info("Nenhum componente identificado com o limiar atual.")

    with tab_json:
        st.json(detections)

    with tab_export:
        json_bytes = json.dumps(detections, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "Descarregar JSON",
            data=json_bytes,
            file_name=f"detections_{uploaded_file.name}.json",
            mime="application/json",
        )

    # ── History ──────────────────────────────────────────────────────────
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "filename": uploaded_file.name,
        "model": selected_folder.name,
        "detections": len(detections),
        "speed_ms": round(result["speed_ms"], 1),
    })

# ── Inference history section ────────────────────────────────────────────────

if st.session_state.history:
    with st.expander(f"Historico de Inferencias ({len(st.session_state.history)})"):
        for entry in reversed(st.session_state.history[-20:]):
            st.text(
                f"[{entry['timestamp']}] {entry['filename']} "
                f"| {entry['model']} "
                f"| {entry['detections']} objetos "
                f"| {entry['speed_ms']} ms"
            )
