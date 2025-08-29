import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Paths
MODEL_PATH = os.path.join("models", "placement_model.pkl")
ENCODER_PATH = os.path.join("models", "encoders.pkl")

# Load model and encoders
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

st.title("🎓 Campus Placement Predictor")

# Inputs
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
interns = st.number_input("Internships", 0, 10, 0)
comm = st.slider("Communication Skill (1-10)", 1, 10, 5)
tech = st.slider("Technical Skills (1-5)", 1, 5, 3)
mock = st.number_input("Mock Test Score", 0, 100, 50)
cert = st.number_input("Certifications", 0, 20, 0)

gender = st.radio("Gender", encoders["Gender"].classes_)
spec = st.selectbox("Specialization", encoders["Specialization"].classes_)
tier = st.selectbox("College Tier", encoders["College_Tier"].classes_)

# Transform categorical inputs
gender_val = encoders["Gender"].transform([gender])[0]
spec_val = encoders["Specialization"].transform([spec])[0]
tier_val = encoders["College_Tier"].transform([tier])[0]

# Match training feature order
features = pd.DataFrame([[
    gender_val, cgpa, spec_val, interns, comm, tech, mock, cert, tier_val
]], columns=["Gender", "CGPA", "Specialization", "Internships", 
             "Communication_Skill", "Technical_Skills", 
             "Mock_Test_Score", "Certifications", "College_Tier"])

# Prediction
if st.button("Predict Placement"):
    result = model.predict(features)[0]
    st.success("🎉 Likely to be Placed!" if result == 1 else "⚠ Unlikely to be Placed.")
