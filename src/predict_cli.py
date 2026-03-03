import joblib
import numpy as np
import pandas as pd

def main():
    model = joblib.load("models/model.joblib")
    imputer = joblib.load("models/imputer.joblib")
    numeric_cols = joblib.load("models/numeric_cols.joblib")

    user_values = []
    print("Enter house features (press Enter to skip and use median):")

    for col in numeric_cols:
        val = input(f"{col}: ")
        if val.strip() == "":
            user_values.append(np.nan)
        else:
            user_values.append(float(val))

    X_input = pd.DataFrame([user_values], columns=numeric_cols)
    X_input_imputed = imputer.transform(X_input)

    predicted_price = model.predict(X_input_imputed)[0]
    print(f"\nPredicted price: {predicted_price:.2f}")

if __name__ == "__main__":
    main()