# ================= IMPORTS =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= LOAD DATA =================
df = pd.read_csv("dataset/crop/Crop_prediction.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.columns)

# ================= CLEAN COLUMN NAMES =================
df.columns = df.columns.str.strip().str.lower()

# ================= CLEAN STATE TEXT =================
df["state"] = df["state"].str.strip().str.title()

print("Unique states:", df["state"].nunique())

# ================= ENCODE STATE =================
state_encoder = LabelEncoder()
df["state_encoded"] = state_encoder.fit_transform(df["state"])
df["soil_encoded"] = LabelEncoder().fit_transform(df["soil_type"])

# ================= EDA =================
plt.figure(figsize=(10,4))
df["crop"].value_counts().plot(kind="bar")
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.show()

# ================= CORRELATION HEATMAP =================
features_for_corr = [
    "n_soil", "p_soil", "k_soil",
    "temperature", "humidity", "ph", "rainfall"
]

plt.figure(figsize=(8,6))
sns.heatmap(df[features_for_corr].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# ================= FEATURES & TARGET =================
X = df[
    ["n_soil","p_soil","k_soil","temperature","humidity","ph",
     "rainfall","state_encoded"]
]


y = df["crop"]

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= MODEL PIPELINE =================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,   # ⭐ THIS IS THE KEY
        C=10,
        gamma="scale"
    ))
])


print("🚀 Training model...")
model.fit(X_train, y_train)

# ================= PREDICTION =================
y_pred = model.predict(X_test)

# ================= METRICS =================
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

plt.figure(figsize=(10,6))
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
plt.title("Crop Prediction Confusion Matrix")
plt.show()

# ================= SAVE MODEL =================
joblib.dump(model, "models/crop_model.pkl")
joblib.dump(state_encoder, "models/state_encoder.pkl")

print("✅ Crop model and state encoder saved successfully!")
