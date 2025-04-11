# scripts/train_and_download.py

import os
import pandas as pd
import seaborn as sns
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------
# 1. Download datasets
# ---------------------------

def download_datasets():
    os.makedirs("data", exist_ok=True)

    print("📥 Downloading Iris dataset...")
    iris = load_iris(as_frame=True)
    iris_df = iris.frame
    iris_df.to_csv("data/iris.csv", index=False)
    print("✅ Saved: data/iris.csv")

    print("📥 Downloading Titanic dataset (from seaborn)...")
    titanic_df = sns.load_dataset("titanic")
    titanic_df.to_csv("data/titanic.csv", index=False)
    print("✅ Saved: data/titanic.csv")

    return iris_df, titanic_df

# ---------------------------
# 2. Preprocessing
# ---------------------------

def preprocess_iris(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42), None

def preprocess_titanic(df):
    df = df.drop(columns=["name", "ticket", "cabin"], errors="ignore")
    df = df.dropna(subset=["embarked", "sex", "age", "fare", "pclass", "survived"])
    df["age"].fillna(df["age"].median(), inplace=True)
    df["fare"].fillna(df["fare"].median(), inplace=True)

    X = df.drop("survived", axis=1)
    y = df["survived"]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

# ---------------------------
# 3. Train and save models
# ---------------------------

def train_models(X_train, y_train, output_dir, preprocessor=None):
    os.makedirs(output_dir, exist_ok=True)

    base_models = {
        "random_forest": RandomForestClassifier(),
        "svm": SVC(probability=True),
        "gb": GradientBoostingClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "lightgbm": LGBMClassifier(),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(),
    }

    for name, model in base_models.items():
        print(f"🔧 Training {name}...")
        if preprocessor:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])
        else:
            pipeline = model  # For Iris: no need for preprocessing

        pipeline.fit(X_train, y_train)
        path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(pipeline, path)
        print(f"✅ Saved: {path}")

# ---------------------------
# 4. Run all steps
# ---------------------------

if __name__ == "__main__":
    iris_df, titanic_df = download_datasets()

    # Iris
    (X_train, X_test, y_train, y_test), _ = preprocess_iris(iris_df)
    train_models(X_train, y_train, "models/iris_models")

    # Titanic
    (X_train, X_test, y_train, y_test), preprocessor = preprocess_titanic(titanic_df)
    train_models(X_train, y_train, "models/titanic_models", preprocessor=preprocessor)
