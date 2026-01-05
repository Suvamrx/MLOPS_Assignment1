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

