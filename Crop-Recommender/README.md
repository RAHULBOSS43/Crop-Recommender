# 🌾 Crop Recommender System

A full-stack machine learning web application that recommends the most suitable crop to cultivate based on soil nutrients, environmental conditions, and optional soil image analysis.

This project combines **Machine Learning + Flask backend + React frontend** to provide an end-to-end intelligent agriculture solution.

---

## 🚀 Project Overview

Selecting the right crop is crucial for maximizing yield and minimizing loss.  
This system analyzes soil and climate parameters and recommends the best crop using trained ML models.

The application is designed with a **clean separation of concerns**:
- ML experiments & training
- Backend REST APIs
- Frontend user interface

---

## ✨ Key Features

- 🌱 Crop recommendation based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - Soil pH
  - Rainfall
- 🖼️ Optional soil image upload for analysis
- 🤖 Machine Learning–based predictions
- 🌐 React + Tailwind modern UI
- 🔌 Flask REST API
- 📦 Modular & scalable project structure

---

## 🏗️ Folder Structure



Crop-Recommender/
│
├── experiments/                  # ML training & testing
│   ├── train_crop_model.py
│   ├── train_soil_model.py
│   ├── test_crop_values.py
│   ├── test_soil_image.py
│   └── predict_full_system.py
│
├── backend/
│   ├── app.py                    # Flask API
│   ├── utils.py
│   ├── models/                   # saved .pkl models
│   ├── dataset/
│   ├── uploads/
│   ├── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── api.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│
├── .gitignore
├── README.md


---

## ⚙️ Tech Stack

### Backend
- Python
- Flask
- Flask-CORS
- Scikit-learn
- NumPy
- Pandas
- OpenCV (for soil image handling)

### Frontend
- React (Vite)
- Tailwind CSS
- JavaScript

### Machine Learning
- Supervised classification models
- Trained on agricultural soil & climate datasets

---

## 🔄 Application Flow

1. User enters soil and environmental parameters via UI
2. Optional soil image is uploaded
3. Frontend sends data to Flask API
4. ML models process inputs
5. Best crop recommendation is generated
6. Result is displayed on the UI

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sunayana90/Crop-Recommender.git
cd Crop-Recommender

---

### 2️⃣ Backend Setup
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py


Backend runs on:

http://127.0.0.1:5000

---

### 3️⃣ Frontend Setup
cd frontend
npm install
npm run dev


Frontend runs on:

http://localhost:5173

---

### 📥 Input Parameters
Parameter	Description
N	Nitrogen content
P	Phosphorus content
K	Potassium content
Temperature	Ambient temperature (°C)
Humidity	Relative humidity (%)
pH	Soil pH value
Rainfall	Rainfall (mm)
📤 Output

✅ Recommended crop best suited for given conditions

---

🧪 Experiments & Training

All model training and testing scripts are stored in archive_experiments/

Models are trained separately and saved as .pkl files

Trained models are loaded dynamically in the backend

---

🔮 Future Enhancements

📍 Location-based automatic soil parameter detection

🌦️ Live weather API integration

🧪 Fertilizer recommendation

📈 Yield prediction

📱 Mobile-responsive UI

🤖 AI chatbot for farmers

---

👩‍💻 Author

Sunayana Yadav
BE EXTC | AI/ML Enthusiast

