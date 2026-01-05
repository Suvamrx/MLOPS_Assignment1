# Project Instructions: Heart Disease Prediction MLOps

This document provides the full technical guide for setting up and running the MLOps pipeline.

## Execution Order
To successfully replicate the project from scratch, run the scripts in this specific order:

1. python scripts/download_Heart_Disease_UCI_Dataset.py: Fetches the raw dataset from the UCI repository.

2. python scripts/train.py: Cleans the data, logs experiments to MLflow, and saves model.joblib.

3. python scripts/inference.py: Verifies the model works locally before deployment.

4. pytest: Runs unit tests to ensure the API logic is sound.

5. kubectl apply -f kubernetes/: Deploys the model to the production cluster.

> **Note:** Always run scripts from the project root directory to ensure file paths are resolved correctly.
---

## 1. Local Environment Setup
Follow these steps to initialize your environment using Conda.

### Setup Conda Environment
```bash
# Create the conda environment with Python 3.9
conda create -n mlops_env python=3.9 -y
```
```bash
# Activate the environment
conda activate mlops_env
```
```bash
#Install Dependencies
# Install the required packages via pip within the conda environment
pip install -r requirements.txt
```
### 2. Data Acquisition
Before running the training pipeline, execute the download script to fetch the raw UCI Heart Disease dataset:
```bash
python scripts/download_data.py
```
---
## 3. Training & Experiment Tracking (MLflow)

The training process is instrumented with MLflow to track hyperparameters and model performance metrics.

**Execute Training:** 
Run the pipeline to process data and train models (Logistic Regression and Random Forest):
```bash
python scripts/train.py
```
**Review Experiments**
Launch the MLflow UI to compare runs:
```bash
mlflow ui
```
* Visit http://localhost:5000 to see the comparison table and accuracy scores.
---

## 4. CI/CD Pipeline (GitHub Actions)

The project uses GitHub Actions to automate code quality and testing.

### Workflow Logic
* **Linting:** Every push triggers **Ruff** to ensure PEP8 compliance.
* **Testing:** **Pytest** runs suite-level tests on API endpoints.
* **Security:** The pipeline only builds the Docker image if all tests pass.

*To view the pipeline status, navigate to the **Actions** tab in your GitHub repository.*

---

## 5. Kubernetes Deployment & Monitoring
The API is deployed as a high-availability service on Kubernetes (Docker Desktop).

### Local Deployment
Apply the manifests to your local cluster:
```bash
# Deploy Pods and Service
kubectl apply -f kubernetes/deployment.yaml
```
### Verification Commands
Check the status of your deployment:
```bash
# Ensure 2/2 pods are 'Running'
kubectl get pods

# Find the External IP/Port for the API
kubectl get svc heart-disease-service
```

### Monitoring (Prometheus Metrics)
The API exposes real-time operational metrics via the Prometheus instrumentator. You can view these by navigating to:
* **Metrics Endpoint:** `http://localhost/metrics`
---

## 6. API Usage Example
You can test the running Kubernetes API using `curl` to ensure it is serving predictions:

```bash
curl -X 'POST' \
  'http://localhost/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 52, 
    "sex": 1, 
    "cp": 0, 
    "trestbps": 125, 
    "chol": 212,
    "fbs": 0, 
    "restecg": 1, 
    "thalach": 168, 
    "exang": 0,
    "oldpeak": 1.0, 
    "slope": 2, 
    "ca": 2, 
    "thal": 3
  }'
  ```
  ---
### 6. Local Inference Testing
To test the model locally without using the Kubernetes API, run the standalone inference script:
```bash
python scripts/inference.py
```