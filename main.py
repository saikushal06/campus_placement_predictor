import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoders.pkl"
DATA_PATH = "data/cleaned_placement_data.csv"

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Initialize encoders for categorical features
    encoders = {}
    for col in ["Gender", "Specialization", "College_Tier"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Prepare features & labels
    X = df.drop("Placed", axis=1)
    y = df["Placed"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("✅ Model trained successfully!")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save encoders
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoders saved to {ENCODER_PATH}")

if __name__ == "__main__":
    train_and_save_model()
