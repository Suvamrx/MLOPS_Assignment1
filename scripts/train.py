import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os

# Set MLflow tracking URI to local directory (works in CI and local environments)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# 1. Load data
df = pd.read_csv('data/raw_heart_disease.csv')
df = df.replace('?', np.nan).fillna(df.median())
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Saving the cleaned dataset
df.to_csv('data/processed_heart_disease.csv', index=False)
print("Cleaned dataset saved to data/processed_heart_disease.csv")

# 2. Define features
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocessing Pipeline (Requirement 4)
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Training Function with MLflow (Requirement 3)
def train_model(model_name, model_obj):
    with mlflow.start_run(run_name=model_name):
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model_obj)])
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std()
        }

        # Generate and log confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        
        # Save plot and log to MLflow
        plot_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)  # Clean up local file after logging

        mlflow.log_params(model_obj.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, "model")
        print(f"Finished {model_name}: {metrics}")
        joblib.dump(clf, 'model.joblib') 
        print("Model saved successfully as model.joblib")

# 5. Run training for two models (Requirement 2)
if __name__ == "__main__":
    train_model("Logistic_Regression", LogisticRegression())
    train_model("Random_Forest", RandomForestClassifier(n_estimators=100))

