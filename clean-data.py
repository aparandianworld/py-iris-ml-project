import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def check_missing_values(df):
    total = df.isnull().sum()
    print(f"missing values: {total}")


def encode_label(df):
    df = df.copy()
    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])
    return df


def split_data(df):
    # Features
    X = df.drop("class", axis=1)
    # Target
    y = df["class"]
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


df = pd.read_csv("data/iris.csv")
check_missing_values(df)
df = encode_label(df)
X_train, X_test, y_train, y_test = split_data(df)
print("Data cleaned and split successfully.")
