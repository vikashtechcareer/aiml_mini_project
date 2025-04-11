

import pandas as pd
import seaborn as sns
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib

# ---------------------------
# 1. Download datasets
# ---------------------------

def download_datasets():
    os.makedirs("data", exist_ok=True)

    print("📥 Downloading Iris dataset...")
    iris = load_iris(as_frame=True)
    iris_df = iris.frame
    iris_df.to_csv("data/iris.csv", index=False)
    print("✅ Iris saved to data/iris.csv")

    print("📥 Downloading Titanic dataset (from seaborn)...")
    titanic_df = sns.load_dataset("titanic")
    titanic_df.to_csv("data/titanic.csv", index=False)
    print("✅ Titanic saved to data/titanic.csv")

    return iris_df, titanic_df

# ---------------------------
# 2. Preprocessing
# ---------------------------

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

# ---------------------------
# 3. Train models
# ---------------------------

def train_models(X_train, y_train, output_dir):
    models = {
        "random_forest": RandomForestClassifier(),
        "svm": SVC(probability=True),
        "gb": GradientBoostingClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "knn": KNeighborsClassifier()
    }

    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        print(f"🔧 Training {name}...")
        model.fit(X_train, y_train)
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"✅ Saved model: {model_path}")

# ---------------------------
# 4. Run everything
# ---------------------------

if __name__ == "__main__":
    iris_df, titanic_df = download_datasets()

    X_train, _, y_train, _ = preprocess_iris(iris_df)
    train_models(X_train, y_train, "models/iris_models")

    X_train, _, y_train, _ = preprocess_titanic(titanic_df)
    train_models(X_train, y_train, "models/titanic_models")
