# scripts/evaluate_models.py
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scripts.model_utils import load_iris, load_titanic
import pandas as pd

def evaluate(dataset_name):
    X_train, X_test, y_train, y_test = (load_iris() if dataset_name == 'iris' else load_titanic())
    report = []
    model_dir = f"models/{dataset_name}_models"

    for model_file in os.listdir(model_dir):
        model_name = model_file.split('.')[0]
        model = joblib.load(f"{model_dir}/{model_file}")
        y_pred = model.predict(X_test)
        report.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1 Score": f1_score(y_test, y_pred, average='macro')
        })
    return report

if __name__ == "__main__":
    iris_report = evaluate("iris")
    titanic_report = evaluate("titanic")
    final = iris_report + titanic_report
    pd.DataFrame(final).to_csv("report/model_performance.csv", index=False)
