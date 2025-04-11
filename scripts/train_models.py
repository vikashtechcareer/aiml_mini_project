# scripts/train_models.py
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scripts.model_utils import load_iris, load_titanic

models = {
    'random_forest': RandomForestClassifier(),
    'svm': SVC(probability=True),
    'gb': GradientBoostingClassifier(),
    'xgboost': XGBClassifier(),
    'lightgbm': LGBMClassifier(),
    'knn': KNeighborsClassifier(),
    'decision_tree': DecisionTreeClassifier()
}

def train_and_save(dataset_name):
    load_fn = load_iris if dataset_name == "iris" else load_titanic
    X_train, X_test, y_train, y_test = load_fn()

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{dataset_name}_models/{name}.joblib")

if __name__ == "__main__":
    train_and_save("iris")
    train_and_save("titanic")
