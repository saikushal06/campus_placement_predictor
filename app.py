import streamlit as st
import pickle
import numpy as np
import os

# Import training function
from main import train_and_save_model

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoders.pkl"

# --- Load model or train if not exists ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.warning("⚠️ No trained model found. Training a new one...")
    train_and_save_model()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

except Exception as e:
    st.error(f"❌ Error loading model or encoders: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("🎓 Campus Placement Predictor")

# User Inputs
cgpa = st.slider("CGPA", 5.0, 10.0, step=0.1)
interns = st.number_input("Internships", 0, 5, step=1)
comm = st.slider("Communication Skill (1-10)", 1, 10, step=1)
tech = st.slider("Technical Skills (1-5)", 1, 5, step=1)
mock = st.number_input("Mock Test Score", 0, 100, step=1)
cert = st.number_input("Certifications", 0, 10, step=1)

gender = st.radio("Gender", ["Male", "Female"])
spec = st.selectbox("Specialization", encoders["Specialization"].classes_)
tier = st.selectbox("College Tier", encoders["College_Tier"].classes_)

# Encode categorical inputs
gender_enc = encoders["Gender"].transform([gender])[0]
spec_enc = encoders["Specialization"].transform([spec])[0]
tier_enc = encoders["College_Tier"].transform([tier])[0]

# Predict button
if st.button("🔮 Predict Placement"):
    features = np.array([[gender_enc, cgpa, spec_enc, interns, comm, tech, mock, cert, tier_enc]])
    result = model.predict(features)[0]

    if result == 1:
        st.success("🎉 The student is **Likely to be Placed!**")
    else:
        st.error("⚠️ The student is **Unlikely to be Placed.**")
