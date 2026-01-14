# ================= IMPORTS =================
import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

# ================= PATHS =================
TRAIN_PATH = "dataset/soil_images/train"
TEST_PATH  = "dataset/soil_images/test"
IMAGE_SIZE = (128, 128)

# ================= DATA LOADER =================
def load_data(base_path):
    """
    Loads images, extracts HOG features,
    and returns feature matrix X and labels y
    """
    X, y = [], []

    for soil_type in os.listdir(base_path):
        class_path = os.path.join(base_path, soil_type)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize to fixed size
            img = cv2.resize(img, IMAGE_SIZE)

            # Convert to grayscale (HOG works on intensity gradients)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 🔥 Optimized HOG (reduced feature size)
            features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(16, 16),   # ⬆ bigger cell → fewer features
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

            X.append(features)
            y.append(soil_type)

    return np.array(X), np.array(y)

# ================= LOAD DATA =================
X_train, y_train = load_data(TRAIN_PATH)
X_test, y_test   = load_data(TEST_PATH)

print("Train samples:", X_train.shape)
print("Test samples:", X_test.shape)

# ================= PIPELINE (FAST + STABLE) =================
# Suppress harmless convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=5,
        max_iter=20000,   # 🔥 increased
        tol=1e-4          # better convergence control
    ))
])

# ================= TRAIN =================
print("\n⏳ Training model...")
model.fit(X_train, y_train)

# ================= PREDICT =================
y_pred = model.predict(X_test)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1          # 🔥 use all CPU cores
)

print(f"Cross-validation accuracy: {scores.mean() * 100:.2f}%")
# ================= METRICS =================
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_test, y_pred)
labels = sorted(set(y_test))

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Soil Type Confusion Matrix")
plt.tight_layout()
plt.show()

# ================= SAVE MODEL =================
joblib.dump(model, "models/soil_svm.pkl")
print("✅ Fast & optimized soil model saved!")
