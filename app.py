from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# 1. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_usage.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. Initialize FastAPI
app = FastAPI(title="Heart Disease Prediction API")

Instrumentator().instrument(app).expose(app)

# 3. Define Input Structure
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

# 4. Load Model
MODEL_PATH = "model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# 5. Endpoints
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is Online"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()
    
    # NEW: Log the prediction for Task 9 (Monitoring)
    result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    logger.info(f"Prediction made: Data={data.dict()}, Result={result_text}, Confidence={max(probability)}")
    
    return {
        "prediction": int(prediction),
        "status": result_text,
        "confidence": max(probability)
    }