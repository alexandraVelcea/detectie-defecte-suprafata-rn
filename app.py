import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import random

# --- CONFIGURARE PAGINA ---
st.set_page_config(page_title="Detector Defecte Suprafata", layout="wide")

# CSS Custom pentru centrarea imaginilor
st.markdown(
    """
    <style>
        [data-testid="stImage"] {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("NEU-DET: Detectie Defecte Suprafata")

# --- SIDEBAR ---
st.sidebar.header("Configurare Model")
conf_threshold = st.sidebar.slider("Prag de Incredere (Confidence)", 0.0, 1.0, 0.25, 0.05)

st.sidebar.divider()
st.sidebar.header("Setari Vizuale")
# Default set to 350px as requested
DISPLAY_WIDTH = st.sidebar.slider("Latime Afisare (px)", 150, 800, 350)
box_thickness = st.sidebar.slider("Grosime Linie", 1, 5, 2)
font_scale = st.sidebar.slider("Dimensiune Text", 0.3, 2.0, 0.6)

# --- INCARCARE MODEL ---
@st.cache_resource
def load_model():
    possible_paths = [
        "models/surface_defect_model/weights/best.pt",
        "yolov8n.pt" 
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return YOLO(path)
    return YOLO("yolov8n.pt")

model = load_model()

def get_color(cls_id):
    random.seed(cls_id)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def resize_for_display(image_input, width):
    """Resizes an image (numpy or PIL) to a target width while keeping aspect ratio."""
    if isinstance(image_input, np.ndarray):
        # OpenCV format (H, W, C)
        h, w = image_input.shape[:2]
        scale = width / w
        return cv2.resize(image_input, (int(width), int(h * scale)))
    else:
        # PIL format
        w_percent = (width / float(image_input.size[0]))
        h_size = int((float(image_input.size[1]) * float(w_percent)))
        return image_input.resize((int(width), h_size), Image.Resampling.LANCZOS)

# --- INCARCARE IMAGINE ---
uploaded_file = st.file_uploader("Alege o imagine...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # 1. Load Original (Full Resolution)
    original_pil = Image.open(uploaded_file)
    
    # 2. Create a smaller copy just for the UI
    display_original = resize_for_display(original_pil, DISPLAY_WIDTH)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagine Originala")
        # Display the resized version
        st.image(display_original, width=DISPLAY_WIDTH)

    # Butonul declanseaza predictia
    if st.sidebar.button("Detecteaza Defecte") or True:
        
        # 3. Predictie (Run on FULL RESOLUTION image for best accuracy)
        results = model.predict(original_pil, conf=conf_threshold)
        result = results[0]

        # 4. Procesare Rezultat
        # Convert full res image to array for drawing
        img_array = np.array(original_pil)
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        annotated_img = img_array.copy()

        # Draw boxes on the FULL resolution image
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = result.names[cls_id]
                color = get_color(cls_id)
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, box_thickness)
                
                label_text = f"{label_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                
                # Draw background for text
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                # Draw text
                cv2.putText(annotated_img, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

            st.toast(f"Detectie completa: {len(boxes)} obiecte gasite.")
        else:
            st.warning("Nu au fost detectate obiecte.")

        # 5. Resize the ANNOTATED image for display
        display_annotated = resize_for_display(annotated_img, DISPLAY_WIDTH)

        with col2:
            st.subheader("Rezultat Vizual")
            st.image(display_annotated, width=DISPLAY_WIDTH)

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