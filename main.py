import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    # Load dataset
    try:
        df = pd.read_csv("dataset.csv")
        print("📂 Dataset loaded successfully!")
        print("Columns in dataset:", df.columns.tolist())
    except Exception as e:
        print("❌ Error loading dataset:", e)
        return

    # Check for 'status' column (target)
    if "status" not in df.columns:
        print("❌ 'status' column not found in dataset. Please check your CSV.")
        return

    # Encode categorical features
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col != "status":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Features and target
    X = df.drop("status", axis=1)
    y = df["status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoders
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print("✅ Model and encoders trained & saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
