import streamlit as st
import pickle
import numpy as np
from main import train_and_save_model

MODEL_PATH = "model.pkl"
ENCODERS_PATH = "encoders.pkl"

# Try loading model and encoders, retrain if not found
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    st.warning("Training model since files not found...")
    train_and_save_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

# App UI
st.title("🎓 Campus Placement Predictor")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
internships = st.number_input("Internships", 0, 10, 0)
comm = st.slider("Communication Skill (1-10)", 1, 10, 5)
tech = st.slider("Technical Skills (1-5)", 1, 5, 3)
mock = st.number_input("Mock Test Score", 0, 100, 0)
cert = st.number_input("Certifications", 0, 10, 0)

# Restrict dropdowns to trained categories
gender = st.selectbox("Gender", encoders["Gender"].classes_.tolist())
specialization = st.selectbox("Specialization", encoders["Specialization"].classes_.tolist())
college_tier = st.selectbox("College Tier", encoders["College_Tier"].classes_.tolist())

if st.button("Predict Placement Chance"):
    try:
        gender_enc = encoders["Gender"].transform([gender])[0]
        specialization_enc = encoders["Specialization"].transform([specialization])[0]
        college_tier_enc = encoders["College_Tier"].transform([college_tier])[0]

        features = np.array([
            cgpa, internships, comm, tech, mock, cert,
            gender_enc, specialization_enc, college_tier_enc
        ]).reshape(1, -1)

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        if pred == 1:
            st.success(f"✅ Likely to be placed! (Confidence: {proba:.2f})")
        else:
            st.error(f"❌ Unlikely to be placed (Confidence: {proba:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

