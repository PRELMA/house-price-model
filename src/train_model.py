import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression 
import joblib

def train_and_save():
    df = pd.read_csv("data/train.csv")
    target_col = "SalePrice"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X_num = X[numeric_cols]

    imputer = SimpleImputer(strategy="median")
    X_num_imputed = imputer.fit_transform(X_num)

    X_train, X_test, y_train, y_test = train_test_split(X_num_imputed, y, test_size=0.2, random_state = 42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.joblib")

