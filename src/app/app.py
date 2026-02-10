import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import json
import os
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NEU-DET Surface Defect Detector", layout="wide")

# Custom CSS for centering
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

st.title("NEU-DET: Surface Defect Detection")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Model Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)

st.sidebar.divider()
st.sidebar.header("Visual Settings")
DISPLAY_WIDTH = st.sidebar.slider("Display Width (px)", 300, 800, 500)
box_thickness = st.sidebar.slider("Box Thickness", 1, 5, 2)
font_scale = st.sidebar.slider("Text Size", 10, 30, 15)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Priority: Augmented Model -> Ultimate Model -> Nano Fallback
    possible_paths = [
        "models/defect_detector_AUG/weights/best.pt",
        "models/defect_detector_ult/weights/best.pt",
        "yolov8n.pt" 
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading model from: {path}")
            return YOLO(path)
    return YOLO("yolov8n.pt")

model = load_model()

# --- LOAD METRICS ---
def load_metrics():
    """
    Loads training history and attempts to get final metrics 
    either from JSON or the last row of the CSV history.
    """
    history_path = "results/training_history.csv"
    metrics_path = "results/test_metrics.json"
    
    df = None
    final_metrics = {}
    
    # 1. Load History CSV
    if os.path.exists(history_path):
        try:
            df = pd.read_csv(history_path)
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            st.error(f"Error loading history CSV: {e}")

    # 2. Load JSON Metrics
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                final_metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics JSON: {e}")
    
    return df, final_metrics

df_history, final_metrics = load_metrics()

# --- UTILS ---
def get_color(cls_id):
    random.seed(cls_id)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def resize_image(image, target_width):
    w_percent = (target_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    return image.resize((target_width, h_size), Image.Resampling.LANCZOS)

# --- MAIN LOGIC ---

# 1. METRICS DASHBOARD (Sidebar)
if final_metrics:
    st.sidebar.divider()
    st.sidebar.subheader("Model Performance")
    
    # --- FIX: MAPPING KEYS FROM YOUR JSON ---
    # Attempt to get 'test_accuracy', fallback to 'accuracy'
    acc = final_metrics.get("test_accuracy", 0.0)
    if acc == 0.0: acc = final_metrics.get("accuracy", 0.0)
        
    # Attempt to get 'test_f1_macro', fallback to 'f1_score'
    f1 = final_metrics.get("test_f1_macro", 0.0)
    if f1 == 0.0: f1 = final_metrics.get("f1_score", 0.0)
    
    col_a, col_b = st.sidebar.columns(2)
    col_a.metric(label="Accuracy", value=f"{acc:.2%}")
    col_b.metric(label="F1 Macro", value=f"{f1:.4f}")
    
    # Optional: Display Latency if available
    latency = final_metrics.get("inference_latency_ms", 0)
    if latency > 0:
        st.sidebar.metric(label="Inference Latency", value=f"{latency} ms")

else:
    st.sidebar.warning("No metrics found in results/test_metrics.json")


# 2. IMAGE UPLOAD & DETECTION
uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    original_pil = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(resize_image(original_pil, DISPLAY_WIDTH))

    # Inference
    results = model.predict(original_pil, conf=conf_threshold)
    result = results[0]
    
    annotated_pil = original_pil.copy()
    draw = ImageDraw.Draw(annotated_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_scale)
    except:
        font = ImageFont.load_default()

    detections = []
    
    if len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]
            color = get_color(cls_id)
            
            # Draw
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)
            
            # Text Label
            text = f"{label} {conf:.0%}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), text, fill="white", font=font)
            
            detections.append({"Type": label, "Confidence": f"{conf:.1%}"})

        with col2:
            st.subheader("Result")
            st.image(resize_image(annotated_pil, DISPLAY_WIDTH))
            st.success(f"Detected {len(detections)} defects.")
            st.table(detections)
    else:
        with col2:
            st.subheader("Result")
            st.image(resize_image(original_pil, DISPLAY_WIDTH))
            st.info("No defects detected.")

# 3. ANALYTICS
st.divider()
st.header("Training Analytics")

if df_history is not None and not df_history.empty:
    tab1, tab2 = st.tabs(["Loss Curves", "Accuracy Curves"])
    
    cols = df_history.columns.tolist()
    
    with tab1:
        st.subheader("Loss")
        loss_cols = [c for c in cols if 'loss' in c.lower()]
        if loss_cols:
            st.line_chart(df_history.set_index('epoch')[loss_cols])
        else:
            st.write("No loss columns found in CSV.")
            
    with tab2:
        st.subheader("Accuracy")
        acc_cols = [c for c in cols if 'map' in c.lower() or 'accuracy' in c.lower()]
        if acc_cols:
            st.line_chart(df_history.set_index('epoch')[acc_cols])
        else:
            st.write("No accuracy columns found in CSV.")