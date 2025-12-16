import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


def load_clean_data():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}")


def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_model.joblib")
    print("Model saved successfully.")


X_train, y_train, X_test, y_test = load_clean_data()
model = train_model(X_train, y_train)
predict_and_evaluate(model, X_test, y_test)
save_model(model)
