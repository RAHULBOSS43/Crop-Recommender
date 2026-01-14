# import cv2
# import joblib
# import pandas as pd
# import numpy as np
# from skimage.feature import hog
# import requests

# # ================= LOAD MODELS =================
# soil_model = joblib.load("models/soil_svm.pkl")
# crop_model = joblib.load("models/crop_model.pkl")
# state_encoder = joblib.load("models/state_encoder.pkl")

# # ================= RULE-BASED SOIL → CROP FILTER =================
# SOIL_CROP_RULES = {
#     "Sandy soil": {"avoid": ["Rice", "Jute", "Paddy"]},
#     "Clay soil": {"avoid": ["Groundnut", "Millet"]},
#     "Black Soil": {"avoid": ["Tea", "Coffee"]},
#     "Alluvial soil": {"avoid": []}
# }

# IMAGE_SIZE = (128, 128)

# print("\n🌱 AI Farming Recommendation System 🌱\n")

# # ================= IMAGE LOADER =================
# def load_image(path_or_url):
#     if path_or_url.startswith("http"):
#         response = requests.get(path_or_url)
#         image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
#         img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
#     else:
#         img = cv2.imread(path_or_url)
#     return img

# # ================= IMAGE INPUT =================
# image_path = input("Enter soil image path or URL: ").strip()
# img = load_image(image_path)

# if img is None:
#     print("❌ Image not found or cannot be loaded.")
#     exit()

# # ================= IMAGE PREPROCESS =================
# img = cv2.resize(img, IMAGE_SIZE)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# features = hog(
#     gray,
#     orientations=9,
#     pixels_per_cell=(16, 16),
#     cells_per_block=(2, 2),
#     block_norm="L2-Hys"
# ).reshape(1, -1)

# # ================= SOIL PREDICTION =================
# soil_type = soil_model.predict(features)[0]
# print(f"\n🌱 Predicted Soil Type: {soil_type}")

# # ================= USER INPUT VALUES =================
# print("\n📋 Enter Soil & Weather Details:")

# n_soil = int(input("Nitrogen (N): "))
# p_soil = int(input("Phosphorus (P): "))
# k_soil = int(input("Potassium (K): "))
# temperature = float(input("Temperature (°C): "))
# humidity = float(input("Humidity (%): "))
# ph = float(input("Soil pH: "))
# rainfall = float(input("Rainfall (mm): "))

# # ================= STATE INPUT (CASE INSENSITIVE) =================
# states = list(state_encoder.classes_)
# states_lower = [s.lower() for s in states]

# user_state = input("\nEnter State Name: ").strip().lower()

# if user_state not in states_lower:
#     print("❌ Invalid state. Available states:")
#     for s in states:
#         print("-", s)
#     exit()

# state_name = states[states_lower.index(user_state)]
# state_encoded = state_encoder.transform([state_name])[0]

# print(f"✅ Selected State: {state_name}")

# # ================= CREATE DATAFRAME =================
# df = pd.DataFrame([[
#     n_soil,
#     p_soil,
#     k_soil,
#     temperature,
#     humidity,
#     ph,
#     rainfall,
#     state_encoded
# ]], columns=[
#     "n_soil",
#     "p_soil",
#     "k_soil",
#     "temperature",
#     "humidity",
#     "ph",
#     "rainfall",
#     "state_encoded"
# ])

# # ================= TOP 3 CROP PREDICTION =================
# probs = crop_model.predict_proba(df)[0]
# crop_names = crop_model.classes_

# crop_prob_pairs = list(zip(crop_names, probs))
# crop_prob_pairs.sort(key=lambda x: x[1], reverse=True)

# top3_crops = crop_prob_pairs[:5]  # take more before filtering

# # ================= APPLY SOIL RULE FILTER =================
# avoid_crops = SOIL_CROP_RULES.get(soil_type, {}).get("avoid", [])

# filtered_crops = [
#     (crop, prob) for crop, prob in top3_crops
#     if crop not in avoid_crops
# ]

# if not filtered_crops:
#     filtered_crops = top3_crops

# # ================= FINAL OUTPUT =================
# print("\n🌾 Top 3 Crop Recommendations:")
# for i, (crop, prob) in enumerate(filtered_crops[:3], start=1):
#     print(f"{i}️⃣ {crop} — {prob*100:.2f}%")

# print("\n✅ Prediction completed successfully!")


import cv2
import joblib
import pandas as pd
import numpy as np
from skimage.feature import hog
import requests

# ================= LOAD MODELS =================
soil_model = joblib.load("models/soil_svm.pkl")
crop_model = joblib.load("models/crop_model.pkl")
state_encoder = joblib.load("models/state_encoder.pkl")

# ================= SOIL → ALLOWED CROPS (STRONG RULES) =================
SOIL_ALLOWED_CROPS = {
    "Sandy soil": ["Bajra", "Groundnut", "Guar", "Millets"],
    "Clay soil": ["Rice", "Wheat", "Banana", "Sugarcane"],
    "Black Soil": ["Cotton", "Soybean", "MothBeans", "Maize", "Blackgram"],
    "Alluvial soil": ["Rice", "Jute", "Wheat", "Sugarcane"]
}

IMAGE_SIZE = (128, 128)

print("\n🌱 AI Farming Recommendation System 🌱\n")

# ================= IMAGE LOADER (PATH OR URL) =================
def load_image(path_or_url):
    try:
        if path_or_url.startswith("http"):
            response = requests.get(path_or_url, timeout=10)
            response.raise_for_status()
            image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path_or_url)
        return img
    except Exception as e:
        print("❌ Error loading image:", e)
        return None

# ================= IMAGE INPUT =================
image_path = input("Enter soil image path or URL: ").strip()
img = load_image(image_path)

if img is None:
    print("❌ Image not found or cannot be loaded.")
    exit()

# ================= IMAGE PREPROCESS =================
img = cv2.resize(img, IMAGE_SIZE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
).reshape(1, -1)

# ================= SOIL PREDICTION =================
soil_type = soil_model.predict(features)[0]
print(f"\n🌱 Predicted Soil Type: {soil_type}")

# ================= USER INPUT VALUES =================
print("\n📋 Enter Soil & Weather Details:")

def get_number(prompt, cast=float):
    while True:
        try:
            return cast(input(prompt))
        except ValueError:
            print("❌ Please enter a valid number.")

n_soil = get_number("Nitrogen (N): ", int)
p_soil = get_number("Phosphorus (P): ", int)
k_soil = get_number("Potassium (K): ", int)
temperature = get_number("Temperature (°C): ")
humidity = get_number("Humidity (%): ")
ph = get_number("Soil pH: ")
rainfall = get_number("Rainfall (mm): ")

# ================= STATE INPUT (CASE INSENSITIVE) =================
states = list(state_encoder.classes_)
states_lower = [s.lower() for s in states]

user_state = input("\nEnter State Name: ").strip().lower()

if user_state not in states_lower:
    print("❌ Invalid state. Available states:")
    for s in states:
        print("-", s)
    exit()

state_name = states[states_lower.index(user_state)]
state_encoded = state_encoder.transform([state_name])[0]

print(f"✅ Selected State: {state_name}")

# ================= CREATE DATAFRAME =================
df = pd.DataFrame([[ 
    n_soil,
    p_soil,
    k_soil,
    temperature,
    humidity,
    ph,
    rainfall,
    state_encoded
]], columns=[
    "n_soil",
    "p_soil",
    "k_soil",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "state_encoded"
])

# ================= CROP PROBABILITIES =================
probs = crop_model.predict_proba(df)[0]
crop_names = crop_model.classes_

crop_prob_pairs = list(zip(crop_names, probs))
crop_prob_pairs.sort(key=lambda x: x[1], reverse=True)

# ================= APPLY SOIL ALLOWED FILTER =================
allowed_crops = SOIL_ALLOWED_CROPS.get(soil_type, [])

# ================= APPLY SOIL ALLOWED FILTER =================
allowed_crops = SOIL_ALLOWED_CROPS.get(soil_type, [])

filtered = [
    (crop, prob) for crop, prob in crop_prob_pairs
    if crop in allowed_crops
]

# 🔥 ENSURE AT LEAST 3 CROPS
if len(filtered) < 3:
    # add highest-probability crops not already included
    for crop, prob in crop_prob_pairs:
        if crop not in [c for c, _ in filtered]:
            filtered.append((crop, prob))
        if len(filtered) == 3:
            break

top3 = filtered[:3]

# ================= FINAL OUTPUT =================
# ================= NORMALIZE PROBABILITIES =================
total_prob = sum(prob for _, prob in top3)

print("\n🌾 Top 3 Crop Recommendations:")
for i, (crop, prob) in enumerate(top3, start=1):
    percent = (prob / total_prob) * 100 if total_prob > 0 else 0
    print(f"{i}️⃣  {crop} — {percent:.2f}%")
