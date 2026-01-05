
import pandas as pd
from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_dataset_exists():
    """Check if the data acquisition script actually downloaded the file."""
    assert os.path.exists("data/raw_heart_disease.csv")

def test_binary_target():
    """Check if the target is actually binary (0 and 1 only)."""
    df = pd.read_csv("data/raw_heart_disease.csv")
    # Apply the same logic used in your training script
    target_values = df['target'].apply(lambda x: 1 if x > 0 else 0).unique()
    assert set(target_values) == {0, 1}

def test_column_count():
    """Ensure all 14 features are present."""
    df = pd.read_csv("data/raw_heart_disease.csv")
    assert df.shape[1] == 14

def test_predict_endpoint():
    response = client.post("/predict", json={
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    })
    assert response.status_code == 200
    assert "prediction" in response.json()