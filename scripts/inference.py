import joblib
import pandas as pd

def run_inference():
    # 1. Load the saved model pipeline
    model = joblib.load('model.joblib')
    
    # 2. Define a sample input (matches the features used in training)
    sample_data = pd.DataFrame([{
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    }])
    
    # 3. Predict
    prediction = model.predict(sample_data)
    probability = model.predict_proba(sample_data)
    
    print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Disease'}")
    print(f"Confidence: {probability[0][prediction[0]]:.2f}")

if __name__ == "__main__":
    run_inference()