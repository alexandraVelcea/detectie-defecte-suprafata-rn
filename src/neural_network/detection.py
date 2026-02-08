from ultralytics import YOLO
import cv2
import sys
import os

def detect_defects(image_path):
    # 1. Load YOUR trained model
    # After training, the best model is saved here. Update path if needed.
    # Check your project root for the 'runs' folder.
    model_path = 'models/surface_defect_model/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first!")
        return

    model = YOLO(model_path)

    # 2. Run Inference
    # conf=0.25 means only show detections with >25% confidence
    results = model.predict(source=image_path, conf=0.25, save=True)

    # 3. Show results
    for result in results:
        # Plot the result (draw boxes)
        res_plotted = result.plot()
        
        # Display using OpenCV
        cv2.imshow("Defect Detection", res_plotted)
        cv2.waitKey(0) # Wait for a key press to close
        cv2.destroyAllWindows()
        
        # Print detected classes
        print(f"Detected {len(result.boxes)} defects:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f" - {model.names[class_id]} ({conf:.2f} confidence)")

if __name__ == '__main__':
    # Usage: python src/neural_network/detection.py path/to/test_image.jpg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        detect_defects(img_path)
    else:
        print("Please provide an image path.")
        print("Example: python src/neural_network/detection.py data/raw/test/some_image.jpg")