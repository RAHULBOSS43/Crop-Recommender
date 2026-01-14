import joblib
import pandas as pd

# ================= LOAD MODEL =================
model = joblib.load("models/crop_model.pkl")
state_encoder = joblib.load("models/state_encoder.pkl")

print("🌾 Crop Recommendation System 🌾\n")

# ================= USER INPUT =================
n_soil = int(input("Enter Nitrogen (N): "))
p_soil = int(input("Enter Phosphorus (P): "))
k_soil = int(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter Soil pH: "))
rainfall = float(input("Enter Rainfall (mm): "))
state = input("Enter State: ")
crop_price = float(input("Enter Crop Price: "))

# ================= ENCODE STATE =================
state_encoded = state_encoder.transform([state.strip().title()])[0]

# ================= CREATE DATAFRAME (ORDER MATTERS) =================
df = pd.DataFrame([[
    n_soil,
    p_soil,
    k_soil,
    temperature,
    humidity,
    ph,
    rainfall,
    state_encoded,
    crop_price
]], columns=[
    "n_soil",
    "p_soil",
    "k_soil",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "state_encoded",
    "crop_price"
])

# ================= PREDICT =================
prediction = model.predict(df)

print("\n🌱 Recommended Crop:", prediction[0])
