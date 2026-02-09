import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import os
import random
import pandas as pd
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NEU-DET Surface Defect Detector", layout="wide")

# Custom CSS for centering images
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
# Changing this slider will now auto-refresh the detection!
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
            return YOLO(path)
    return YOLO("yolov8n.pt")

model = load_model()

# --- LOAD METRICS ---
def load_metrics():
    """Loads training history and final metrics from results folder."""
    history_path = "results/training_history.csv"
    metrics_path = "results/test_metrics.json"
    
    df = None
    final_metrics = None
    
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        # Rename columns for cleaner display if necessary
        df.columns = [c.strip() for c in df.columns]

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            final_metrics = json.load(f)
            
    return df, final_metrics

df_history, final_metrics = load_metrics()

# --- UTILS ---
def get_color(cls_id):
    random.seed(cls_id)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def resize_image(image, target_width):
    """Resizes PIL image to target width while maintaining aspect ratio."""
    w_percent = (target_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    return image.resize((target_width, h_size), Image.Resampling.LANCZOS)

# --- MAIN APP LOGIC ---

# 1. METRICS DASHBOARD (Sidebar)
if final_metrics:
    st.sidebar.divider()
    st.sidebar.subheader("Model Accuracy")
    
    # Accuracy (mAP@50)
    acc = final_metrics.get("accuracy", 0.0)
    st.sidebar.metric(label="Accuracy (mAP@50)", value=f"{acc:.1%}")
    
    # F1 Score
    f1 = final_metrics.get("f1_score", 0.0)
    st.sidebar.metric(label="F1 Score", value=f"{f1:.4f}")

# 2. IMAGE UPLOAD & AUTOMATIC DETECTION
uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Load and Resize for Display
    original_pil = Image.open(uploaded_file).convert("RGB")
    display_img = resize_image(original_pil, DISPLAY_WIDTH)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(display_img)

    # --- AUTOMATIC DETECTION START ---
    # No button needed. This code runs immediately when file is uploaded.
    
    # Run Inference
    results = model.predict(original_pil, conf=conf_threshold)
    result = results[0]

    # Prepare Annotation Canvas
    annotated_pil = original_pil.copy()
    draw = ImageDraw.Draw(annotated_pil)
    
    # Load font
    try:
        # Try a standard font, fallback to default if missing
        font = ImageFont.truetype("arial.ttf", font_scale)
    except IOError:
        font = ImageFont.load_default()

    boxes = result.boxes
    detections_data = []

    if len(boxes) > 0:
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Class and Confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_name = result.names[cls_id]
            color = get_color(cls_id)
            
            # Draw Box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)
            
            # Label Text
            label_text = f"{label_name} {conf:.0%}"
            
            # Calculate Text Size for background
            left, top, right, bottom = draw.textbbox((x1, y1), label_text, font=font)
            text_width = right - left
            text_height = bottom - top
            
            # Adjust label position (above box, or inside if too high)
            text_y = y1 - text_height - 5
            if text_y < 0: 
                text_y = y1 + 5
            
            # Draw Text Background
            draw.rectangle(
                [x1, text_y, x1 + text_width + 4, text_y + text_height + 4],
                fill=color
            )
            # Draw Text
            draw.text((x1 + 2, text_y), label_text, fill="white", font=font)

            # Add to data table
            detections_data.append({
                "Type": label_name,
                "Confidence": f"{conf:.2%}",
                "Location": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            })

        # Display Result in Column 2
        display_annotated = resize_image(annotated_pil, DISPLAY_WIDTH)
        with col2:
            st.subheader("Defect Detection")
            st.image(display_annotated)
        
        # Display Success Message & Table
        st.success(f"Detected {len(boxes)} defects.")
        st.dataframe(detections_data, use_container_width=True)

    else:
        # No defects found
        with col2:
            st.subheader("Result")
            st.image(display_img) # Show clean image
            st.info("No defects detected (Clean Surface).")

# --- 3. TRAINING ANALYTICS SECTION ---
st.divider()
st.header("Training Analytics")

if df_history is not None:
    tab1, tab2 = st.tabs(["Error Graphs (Loss)", "Accuracy Curves"])

    with tab1:
        st.subheader("Training vs Validation Loss")
        st.write("Lower values indicate better performance.")
        loss_data = df_history[['epoch', 'train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss']].set_index('epoch')
        st.line_chart(loss_data)

    with tab2:
        st.subheader("Accuracy (mAP) over Epochs")
        st.write("Higher values indicate better accuracy.")
        acc_data = df_history[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].set_index('epoch')
        st.line_chart(acc_data)
else:
    st.warning("Training history not found. Run 'export_history.py' first.")