import streamlit as st
import pickle
import os
from main import train_and_save_model, MODEL_PATH, ENCODERS_PATH

# Check if model exists, else train
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    train_and_save_model()

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load encoders
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

st.title("🎓 Campus Placement Predictor")

# Example input fields (customize as per dataset columns)
gender = st.selectbox("Gender", ["M", "F"])
ssc_p = st.slider("SSC Percentage", 0, 100, 60)
hsc_p = st.slider("HSC Percentage", 0, 100, 60)
degree_p = st.slider("Degree Percentage", 0, 100, 60)

# Apply encoders
gender_encoded = encoders["gender"].transform([gender])[0]

# Create feature vector (⚠️ must match training order exactly)
features = [[gender_encoded, ssc_p, hsc_p, degree_p]]

if st.button("Predict Placement"):
    prediction = model.predict(features)[0]
    result = "✅ Placed" if prediction == 1 else "❌ Not Placed"
    st.success(f"Prediction: {result}")
