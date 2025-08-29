import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths
DATA_PATH = os.path.join("data", "cleaned_placement_data.csv")
MODEL_PATH = os.path.join("models", "placement_model.pkl")
ENCODER_PATH = os.path.join("models", "encoders.pkl")

# Load data
df = pd.read_csv(DATA_PATH)

# Encode categorical variables and store encoders
encoders = {}
for col in ["Gender", "Specialization", "College_Tier"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split data
X = df.drop("Placed", axis=1)
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model and encoders
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoders, f)

print("✅ Model and encoders saved successfully!")
