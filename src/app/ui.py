import sys
import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from ultralytics import YOLO

# ---------- DUMMY UI ----------

# --- CONFIGURATION ---
# Resolve paths relative to this script
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "defect_detector" / "weights" / "best.pt"

class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Surface Defect Detection System (SIA)")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.image_path = None
        self.original_image = None
        self.current_image = None
        self.model = None

        # --- GUI LAYOUT ---
        
        # 1. Header Section
        header_frame = tk.Frame(root, bg="#2c3e50", height=60)
        header_frame.pack(fill="x")
        
        title_label = tk.Label(
            header_frame, 
            text="Industrial Surface Defect Detection", 
            font=("Arial", 18, "bold"), 
            bg="#2c3e50", 
            fg="white"
        )
        title_label.pack(pady=15)

        # 2. Control Panel (Buttons)
        control_frame = tk.Frame(root, bg="#ecf0f1", pady=10)
        control_frame.pack(fill="x")

        self.btn_load = tk.Button(
            control_frame, text="Load Image", command=self.load_image,
            font=("Arial", 12), bg="#3498db", fg="white", width=15
        )
        self.btn_load.pack(side="left", padx=20)

        self.btn_detect = tk.Button(
            control_frame, text="Detect Defects", command=self.detect_defects,
            font=("Arial", 12), bg="#e74c3c", fg="white", width=15,
            state="disabled" # Disabled until image is loaded
        )
        self.btn_detect.pack(side="left", padx=20)

        self.status_label = tk.Label(control_frame, text="Status: Ready", bg="#ecf0f1", font=("Arial", 10))
        self.status_label.pack(side="right", padx=20)

        # 3. Main Content Area
        content_frame = tk.Frame(root, bg="#f0f0f0")
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: Image Display
        self.image_panel = tk.Label(content_frame, text="No Image Loaded", bg="#bdc3c7", relief="sunken")
        self.image_panel.pack(side="left", fill="both", expand=True, padx=5)

        # Right: Results Panel
        results_frame = tk.Frame(content_frame, bg="white", width=250, relief="sunken", borderwidth=1)
        results_frame.pack(side="right", fill="y", padx=5)
        
        results_title = tk.Label(results_frame, text="Detection Results", font=("Arial", 12, "bold"), bg="white")
        results_title.pack(pady=10)

        self.results_text = tk.Text(results_frame, height=20, width=30, bg="white", borderwidth=0, font=("Consolas", 10))
        self.results_text.pack(padx=10, pady=5)
        self.results_text.insert("end", "Waiting for detection...")
        self.results_text.config(state="disabled")

        # --- INITIALIZATION ---
        self.load_model()

    def load_model(self):
        """Loads the YOLO model on startup."""
        if not MODEL_PATH.exists():
            messagebox.showerror("Error", f"Model not found at:\n{MODEL_PATH}\n\nPlease run training first!")
            self.status_label.config(text="Status: Model Missing", fg="red")
            return
        
        try:
            self.status_label.config(text="Status: Loading Model...", fg="blue")
            self.root.update()
            self.model = YOLO(MODEL_PATH)
            self.status_label.config(text="Status: Model Loaded", fg="green")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")

    def load_image(self):
        """Opens file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.status_label.config(text=f"Loaded: {Path(file_path).name}", fg="black")
            
            # Display Original
            img = Image.open(file_path)
            self.original_image = img
            self.display_image(img)
            
            # Reset UI
            self.btn_detect.config(state="normal")
            self.update_results_text("Image loaded.\nClick 'Detect' to start.")

    def detect_defects(self):
        """Runs inference on the loaded image."""
        if not self.model:
            messagebox.showwarning("Warning", "Model is not loaded.")
            return

        self.status_label.config(text="Status: Processing...", fg="orange")
        self.root.update()

        try:
            # Run YOLO inference
            results = self.model.predict(source=self.image_path, conf=0.25)
            result = results[0]

            # 1. Get the visual result (plotted image)
            # YOLO returns BGR numpy array, convert to RGB for Pillow
            res_array = result.plot() 
            res_image = Image.fromarray(cv2.cvtColor(res_array, cv2.COLOR_BGR2RGB))
            
            # 2. Display the result
            self.display_image(res_image)

            # 3. Parse text results
            summary = "--- DETECTIONS ---\n\n"
            if len(result.boxes) == 0:
                summary += "No defects detected.\n(Clean Surface)"
            else:
                counts = {}
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    
                    summary += f"- {label.upper()}: {conf:.1%}\n"
                    counts[label] = counts.get(label, 0) + 1
                
                summary += "\n--- SUMMARY ---\n"
                for label, count in counts.items():
                    summary += f"{label}: {count}\n"

            self.update_results_text(summary)
            self.status_label.config(text="Status: Detection Complete", fg="green")

        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
            self.status_label.config(text="Status: Error", fg="red")

    def display_image(self, pil_image):
        """Resizes and displays image in the GUI."""
        # Calculate aspect ratio to fit in panel (approx 600x500)
        panel_w = 600
        panel_h = 500
        
        ratio_w = panel_w / pil_image.width
        ratio_h = panel_h / pil_image.height
        scale = min(ratio_w, ratio_h)
        
        new_w = int(pil_image.width * scale)
        new_h = int(pil_image.height * scale)
        
        img_resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        
        self.image_panel.config(image=self.tk_image, text="")
        self.image_panel.image = self.tk_image

    def update_results_text(self, text):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", text)
        self.results_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = DefectDetectionApp(root)
    root.mainloop()