# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import joblib
# import requests
# import pandas as pd
# from skimage.feature import hog

# app = Flask(__name__)
# CORS(app)

# # ===== LOAD MODELS =====
# soil_model = joblib.load("models/soil_svm.pkl")
# crop_model = joblib.load("models/crop_model.pkl")
# state_encoder = joblib.load("models/state_encoder.pkl")

# IMAGE_SIZE = (128, 128)

# SOIL_CROP_RULES = {
#     "Sandy soil": ["Rice", "Jute", "Paddy"],
#     "Clay soil": ["Groundnut", "Millet"],
#     "Black Soil": ["Tea", "Coffee"],
#     "Alluvial soil": []
# }

# # ===== HELPER =====
# def load_image_from_url(url):
#     resp = requests.get(url, timeout=10)
#     img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
#     return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

# # ===== ROUTES =====

# @app.route("/states", methods=["GET"])
# def get_states():
#     return jsonify(sorted(list(state_encoder.classes_)))

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json

#     image_url = data.get("image_url")
#     if not image_url:
#         return jsonify({"error": "Image URL required"}), 400

#     img = load_image_from_url(image_url)
#     if img is None:
#         return jsonify({"error": "Image could not be loaded"}), 400

#     img = cv2.resize(img, IMAGE_SIZE)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     features = hog(
#         gray,
#         orientations=9,
#         pixels_per_cell=(16, 16),
#         cells_per_block=(2, 2),
#         block_norm="L2-Hys"
#     ).reshape(1, -1)

#     soil_type = soil_model.predict(features)[0]

#     # ===== INPUT VALUES =====
#     df = pd.DataFrame([[
#         data["n"],
#         data["p"],
#         data["k"],
#         data["temperature"],
#         data["humidity"],
#         data["ph"],
#         data["rainfall"],
#         state_encoder.transform([data["state"]])[0]
#     ]], columns=[
#         "n_soil","p_soil","k_soil",
#         "temperature","humidity","ph",
#         "rainfall","state_encoded"
#     ])

#     probs = crop_model.predict_proba(df)[0]
#     crops = crop_model.classes_

#     pairs = list(zip(crops, probs))
#     pairs.sort(key=lambda x: x[1], reverse=True)

#     avoid = SOIL_CROP_RULES.get(soil_type, [])
#     filtered = [(c,p) for c,p in pairs if c not in avoid][:3]

#     total = sum(p for _,p in filtered)
#     results = [
#         {"crop": c, "confidence": round((p/total)*100,2)}
#         for c,p in filtered
#     ]

#     return jsonify({
#         "soil": soil_type,
#         "recommendations": results
#     })

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, numpy as np, joblib, requests, os
from skimage.feature import hog
import pandas as pd

app = Flask(__name__)

# ✅ FIX CORS (React ↔ Flask)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODELS =================
soil_model = joblib.load("models/soil_svm.pkl")
crop_model = joblib.load("models/crop_model.pkl")
state_encoder = joblib.load("models/state_encoder.pkl")

IMAGE_SIZE = (128, 128)

# ================= IMAGE LOADER =================
def load_image(file=None, url=None):
    try:
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            img = cv2.imread(path)
            return img

        if url:
            r = requests.get(url, timeout=10)
            img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    except:
        return None

    return None


# ================= STATES API =================
@app.route("/states", methods=["GET"])
def get_states():
    return jsonify(list(state_encoder.classes_))


# ================= PREDICTION API =================
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    url = request.form.get("image_url")

    # ✅ Image validation
    if not file and not url:
        return jsonify({"error": "Image or URL required"}), 400

    img = load_image(file, url)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # ================= IMAGE PROCESS =================
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    ).reshape(1, -1)

    soil = soil_model.predict(features)[0]

    # ================= FORM DATA =================
    data = request.form

    required = ["n", "p", "k", "temp", "humidity", "ph", "rainfall", "state"]
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"}), 400

    if data["state"] not in state_encoder.classes_:
        return jsonify({"error": "Invalid state"}), 400

    state_encoded = state_encoder.transform([data["state"]])[0]

    df = pd.DataFrame([[
        int(data["n"]),
        int(data["p"]),
        int(data["k"]),
        float(data["temp"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"]),
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

    # ================= CROP PREDICTION =================
    probs = crop_model.predict_proba(df)[0]
    crops = crop_model.classes_

    top3 = sorted(
        zip(crops, probs),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return jsonify({
        "soil": soil,
        "crops": [
            {"name": c, "percent": round(p * 100, 2)}
            for c, p in top3
        ]
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
