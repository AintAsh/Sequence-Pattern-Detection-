import os
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ---------- Load model & scaler ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "pattern_rnn_model.h5")
scaler_path = os.path.join(BASE_DIR, "scaler_pattern.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# ---------- FastAPI app ----------
app = FastAPI(title="Pattern Detection API")

# ---------- Input schema ----------
class SequenceInput(BaseModel):
    sequence: list[int]  # expects 5 numbers

# ---------- Root endpoint ----------
@app.get("/")
def home():
    return {"message": "Pattern Detection API is running"}

# ---------- Prediction endpoint ----------
@app.post("/predict")
def predict_pattern(data: SequenceInput):
    seq = np.array(data.sequence).reshape(1, 5)

    seq_scaled = scaler.transform(seq)
    seq_scaled = seq_scaled.reshape(1, 5, 1)

    prob = float(model.predict(seq_scaled)[0][0])
    prediction = 1 if prob >= 0.5 else 0

    return {
        "prediction": prediction,
        "probability": prob
    }
