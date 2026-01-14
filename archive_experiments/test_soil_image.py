import cv2
import joblib
import numpy as np
from skimage.feature import hog

# ================= LOAD MODEL =================
model = joblib.load("models/soil_svm.pkl")

IMAGE_SIZE = (128, 128)

# ================= LOAD IMAGE =================
image_path = "test_images/Clay_1.jpg"
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

# Resize & preprocess
img = cv2.resize(img, IMAGE_SIZE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extract HOG features (SAME AS TRAINING)
features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)

# Convert to model input shape
features = features.reshape(1, -1)

# ================= PREDICT =================
prediction = model.predict(features)

print("🌱 Predicted Soil Type:", prediction[0])
