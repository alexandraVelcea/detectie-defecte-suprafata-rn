import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import random

# --- CONFIGURARE PAGINA ---
st.set_page_config(page_title="Detector Defecte Suprafata", layout="wide")

st.title("NEU-DET: Detectie Defecte Suprafata")

# --- SIDEBAR ---
st.sidebar.header("Configurare Model")
conf_threshold = st.sidebar.slider("Prag de Incredere (Confidence)", 0.0, 1.0, 0.25, 0.05)

st.sidebar.divider()
st.sidebar.header("Setari Vizuale")
box_thickness = st.sidebar.slider("Grosime Linie", 1, 5, 2)
font_scale = st.sidebar.slider("Dimensiune Text", 0.3, 2.0, 0.6)

# --- INCARCARE MODEL ---
@st.cache_resource
def load_model():
    # Exemplu de cai posibile
    possible_paths = [
        "runs/detect/defect_detector_HD/weights/best.pt",
        "yolov8n.pt" # Fallback pentru testare rapida daca nu ai modelul custom
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return YOLO(path)
    
    # Daca nu gaseste nimic local, descarca yolov8n standard (doar pentru demo)
    return YOLO("yolov8n.pt")

model = load_model()

# Functie pentru generarea de culori consistente pe baza ID-ului clasei
def get_color(cls_id):
    random.seed(cls_id)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# --- INCARCARE IMAGINE ---
uploaded_file = st.file_uploader("Alege o imagine...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Convertim imaginea PIL in format OpenCV (numpy array)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagine Originala")
        st.image(image, use_column_width=True)

    # Butonul declanseaza predictia
    if st.sidebar.button("Detecteaza Defecte") or True:
        
        # Predictie
        results = model.predict(image, conf=conf_threshold)
        result = results[0]

        # Copiem imaginea pentru a desena pe ea
        # Daca imaginea are canal Alpha (RGBA), o convertim la RGB
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        annotated_img = img_array.copy()

        # Iteram prin fiecare cutie (box) detectata
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                # 1. Extrage coordonatele (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 2. Extrage clasa si increderea
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = result.names[cls_id]
                
                # 3. Determina culoarea (RGB)
                color = get_color(cls_id)
                
                # 4. Deseneaza dreptunghiul
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, box_thickness)
                
                # 5. Deseneaza eticheta (Text)
                label_text = f"{label_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                
                # Fundal pentru text (pentru lizibilitate)
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                # Textul propriu-zis (Alb)
                cv2.putText(annotated_img, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

            st.toast(f"Detectie completa: {len(boxes)} obiecte gasite.")
        else:
            st.warning("Nu au fost detectate obiecte.")

        with col2:
            st.subheader("Rezultat Vizual")
            st.image(annotated_img, use_column_width=True)

        # Tabel date
        if len(boxes) > 0:
            st.divider()
            st.subheader("Date Detectie")
            data = []
            for box in boxes:
                row = {
                    "Clasa": result.names[int(box.cls[0])],
                    "Incredere": f"{float(box.conf[0]):.2%}",
                    "Coordonate": f"[{int(box.xyxy[0][0])}, {int(box.xyxy[0][1])} ...]"
                }
                data.append(row)
            st.dataframe(data, use_container_width=True)