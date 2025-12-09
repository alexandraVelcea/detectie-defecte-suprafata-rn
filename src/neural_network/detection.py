from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Load YOUR trained model
# Make sure the path points to where the training script saved the weights
model = YOLO('runs/detect/surface_defect_model/weights/best.pt')

# 2. Run inference on a test image
# Replace this with the path to an image from your test set or a new image
image_path = 'datasets/NEU-DET/valid/images/scratches_100.jpg' 

results = model.predict(
    source=image_path,
    conf=0.25,      # Minimum confidence threshold (25%)
    save=True       # Save the image with boxes drawn
)

# 3. Display the result directly in Python
# results[0].plot() creates a numpy array of the image with boxes
result_img = results[0].plot()

# Convert Color (OpenCV uses BGR, Matplotlib uses RGB)
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

plt.imshow(result_img)
plt.axis('off')
plt.show()