import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json
import os
from pathlib import Path
from ultralytics import YOLO

MODELS_DIR = Path("../modelos")

def get_available_models():
    if not MODELS_DIR.exists():
        return []
    models = []
    for pt_file in MODELS_DIR.rglob("*.pt"):
        models.append(str(pt_file))
    return models

st.set_page_config(page_title="Wheel Inspector", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: none;
    }
    .stImage > img {
        border-radius: 8px;
        border: 1px solid #3e4451;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Wheel Inspector")
st.markdown("Plataforma de Inspeção Visual para Deteção de Rodas, Jantes e Parafusos.")

st.sidebar.header("Configuração de Inspeção")
available_models = get_available_models()

if not available_models:
    st.sidebar.error("Nenhum modelo (.pt) encontrado na pasta `modelos/`.")
else:
    selected_model_path = st.sidebar.selectbox("Ficheiro de Pesos", available_models)
    conf_threshold = st.sidebar.slider("Confiança de Deteção", 0.0, 1.0, 0.25, 0.05)
    
    @st.cache_resource
    def load_model(path):
        return YOLO(path)
        
    try:
        model = load_model(selected_model_path)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        model = None

    st.divider()
    
    uploaded_file = st.file_uploader("Arraste ou selecione a imagem para inspeção técnica", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('A processar rede neuronal...'):
            results = model.predict(source=image, conf=conf_threshold, verbose=False)
            result = results[0]
            
            # Formatação de dados
            detections = []
            for box in result.boxes:
                detections.append({
                    "Classe": model.names[int(box.cls)],
                    "Confiança": f"{float(box.conf):.2%}"
                })
            
            annotated_image = result.plot()

        m1, m2, m3 = st.columns(3)
        m1.metric("Objetos Detetados", len(detections))
        m2.metric("Tempo Processamento", f"{result.speed['inference']:.1f}ms")
        m3.metric("Resolução Input", f"{image.shape[1]}x{image.shape[0]}")

        st.divider()

        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("### Imagem Original")
            st.image(image, channels="BGR", width="stretch")
            
        with col_img2:
            st.markdown("### Análise de Bounding Boxes")
            st.image(annotated_image, channels="BGR", width="stretch")

        st.divider()

        tab_list, tab_json = st.tabs(["Lista de Componentes", "Output Técnico (JSON)"])
        
        with tab_list:
            if detections:
                st.table(detections)
            else:
                st.info("Nenhum componente identificado com o limiar de confiança atual.")
                
        with tab_json:
            json_str = result.to_json()
            st.json(json.loads(json_str))