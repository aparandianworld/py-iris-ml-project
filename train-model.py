import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import os


def load_clean_data():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    return X_train, y_train, X_test, y_test


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}")
    return accuracy


def save_model(model, filename):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{filename}")
    print("Model saved successfully.")


accuracies = {}
models = {
    "RandomForestClassifier": RandomForestClassifier(),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
}

X_train, y_train, X_test, y_test = load_clean_data()
for name, model in models.items():
    trained_model = train_model(model, X_train, y_train)
    accuracy = predict_and_evaluate(trained_model, X_test, y_test)
    accuracies[name] = accuracy
    save_model(trained_model, f"{name}_iris_model.joblib")

print("All models trained and saved successfully.")
print("\n Model comparison: ")
for name, accuracy in accuracies.items():
    print(f"{name}: {accuracy * 100:.2f}")
