import pandas as pd
import pytest
import os

# Data file paths
DATA_PATH = "data/raw_heart_disease.csv"
MODEL_PATH = "model.joblib"


# ==================== Data Tests ====================

def test_dataset_exists():
    """Check if the data acquisition script actually downloaded the file."""
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Dataset not available")
def test_binary_target():
    """Check if the target is actually binary (0 and 1 only)."""
    df = pd.read_csv(DATA_PATH)
    target_values = df['target'].apply(lambda x: 1 if x > 0 else 0).unique()
    assert set(target_values) == {0, 1}


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Dataset not available")
def test_column_count():
    """Ensure all 14 features are present."""
    df = pd.read_csv(DATA_PATH)
    assert df.shape[1] == 14


# ==================== API Tests ====================

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not available")
def test_predict_endpoint():
    """Test the /predict endpoint returns valid predictions."""
    from fastapi.testclient import TestClient
    from app import app
    
    client = TestClient(app)
    response = client.post("/predict", json={
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "status" in response.json()
    assert "confidence" in response.json()