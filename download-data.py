import requests
import pandas as pd
import os


def preview_data(df):
    print(df.head())


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

os.makedirs("data", exist_ok=True)

try:
    if os.path.exists("data/iris.csv"):
        print("Data exists in local file system.")

    else:
        response = requests.get(url)
        response.raise_for_status()

        if response.status_code == 200:
            with open("data/iris.csv", "w") as fh:
                fh.write(response.text)

        df = pd.read_csv("data/iris.csv", header=None, names=column_names)
        preview_data(df)
        df.to_csv("data/iris.csv", index=False)
        print("Data downloaded and saved successfully.")

except requests.RequestException as e:
    print(f"Error downloading data: {e}")

except OSError as e:
    print(f"Error with file operations: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
