# calibration/ml_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from config import ML_MODEL_PATH

def train_model(csv_path="calibration/shot_log.csv"):
    df = pd.read_csv(csv_path)

    # Convert spin to one-hot encoding
    df = pd.get_dummies(df, columns=["spin"], prefix="spin")

    # Input features: angle, speed, spin
    features = ["angle", "speed"] + [col for col in df.columns if col.startswith("spin_")]
    X = df[features]

    # Targets: predicted vs real object ball position (delta X and Y)
    y = df[["obj_x", "obj_y"]]  # future: change to delta if needed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    print(f"Model R² score on test set: {model.score(X_test, y_test):.4f}")

    joblib.dump(model, ML_MODEL_PATH)
    print(f"Model saved to {ML_MODEL_PATH}")

def predict_correction(input_data):
    import os
    if not os.path.exists(ML_MODEL_PATH):
        print("No trained model found.")
        return (0, 0)

    model = joblib.load(ML_MODEL_PATH)
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    for col in ["spin_back", "spin_left", "spin_none", "spin_right", "spin_top"]:
        if col not in df.columns:
            df[col] = 0
    return model.predict(df)[0]
