# scripts/evaluate_models.py

import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------- Preprocessing ----------

def preprocess_iris(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_titanic(df):
    df = df.drop(columns=["name", "ticket", "cabin"], errors="ignore")
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df = df.dropna()

    X = df.drop("survived", axis=1)
    y = df["survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Evaluation ----------

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

def evaluate_all_models(model_dir, X_test, y_test):
    results = []
    for file in os.listdir(model_dir):
        if file.endswith(".joblib"):
            model_path = os.path.join(model_dir, file)
            model_name = file.replace(".joblib", "")
            model = joblib.load(model_path)
            print(f"🔍 Evaluating: {model_name}")
            metrics = evaluate_model(model_name, model, X_test, y_test)
            results.append(metrics)
    return results

# ---------- Main Script ----------

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    # Load Iris data
    print("📊 Evaluating Iris models...")
    iris = load_iris(as_frame=True)
    iris_df = iris.frame
    X_train, X_test, y_train, y_test = preprocess_iris(iris_df)
    iris_results = evaluate_all_models("models/iris_models", X_test, y_test)
    iris_df_result = pd.DataFrame(iris_results)
    iris_df_result.to_csv("reports/iris_model_report.csv", index=False)
    print("✅ Saved Iris report to reports/iris_model_report.csv")

    # Load Titanic data
    print("📊 Evaluating Titanic models...")
    titanic_df = sns.load_dataset("titanic")
    X_train, X_test, y_train, y_test = preprocess_titanic(titanic_df)
    titanic_results = evaluate_all_models("models/titanic_models", X_test, y_test)
    titanic_df_result = pd.DataFrame(titanic_results)
    titanic_df_result.to_csv("reports/titanic_model_report.csv", index=False)
    print("✅ Saved Titanic report to reports/titanic_model_report.csv")
