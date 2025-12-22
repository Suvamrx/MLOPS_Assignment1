import pandas as pd
import os

def download_heart_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    
    # Download and load data
    df = pd.read_csv(url, names=columns, na_values="?")
    
    # Save to data directory
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_heart_disease.csv', index=False)
    print("Dataset downloaded and saved to data/raw_heart_disease.csv")

if __name__ == "__main__":
    download_heart_data()