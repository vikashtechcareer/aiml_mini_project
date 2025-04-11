# scripts/model_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_iris():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame
    X = df.drop(columns='target')
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_titanic():
    df = pd.read_csv("data/titanic.csv")
    df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)
