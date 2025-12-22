from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Heart Disease Prediction API")

# 1. Load the best model (Adjust path if needed)
# Ensure you have 'model.joblib' or use MLflow to export it
MODEL_PATH = "model.joblib"

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# 2. Define the input data structure (Pydantic)
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is Online"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame for the model
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()
    
    return {
        "prediction": int(prediction),
        "status": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "confidence": probability
    }