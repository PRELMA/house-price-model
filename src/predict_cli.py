import joblib
import numpy as np

def main():
    model = joblib.load("models/model.joblib")
    imputer = joblib.load("models/imputer.joblib")
    numeric_cols = joblib.load("models/nuneric_cols.joblib")

    user_values = []
    print("Enter house features (press Enter to skip and use median):")

    