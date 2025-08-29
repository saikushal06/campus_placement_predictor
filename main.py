import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "placement_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

def train_and_save_model():
    # Load dataset
    df = pd.read_csv("dataset/placement.csv")

    # Encode categorical columns
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split features/target
    X = df.drop("status", axis=1)
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Create model directory if missing
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save encoders
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print("✅ Model and encoders trained & saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
