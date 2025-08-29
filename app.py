import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or encoders: {e}")
    st.stop()

st.title("🎓 Campus Placement Predictor")

# Example input fields (update according to your dataset columns)
inputs = {}

st.subheader("Enter Student Details")

# Add inputs dynamically from encoders
for col in encoders.keys():
    options = encoders[col].classes_
    inputs[col] = st.selectbox(f"{col}", options)

# For numeric columns (not encoded)
numeric_features = ["cgpa", "iq", "resume_score"]  # change as per your dataset
for col in numeric_features:
    inputs[col] = st.number_input(f"{col}", min_value=0.0, max_value=100.0, step=0.1)

# Prepare input for prediction
if st.button("Predict Placement"):
    input_df = pd.DataFrame([inputs])

    # Encode categorical columns
    for col, le in encoders.items():
        input_df[col] = le.transform([input_df[col][0]])

    # Predict
    prediction = model.predict(input_df)[0]
    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.success("✅ The student is likely to get placed!")
    else:
        st.error("❌ The student might not get placed.")
