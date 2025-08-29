import streamlit as st
import pickle
import numpy as np
import os

# Load model safely
model_path = os.path.join("models", "placement_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Campus Placement Predictor")

# Inputs
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
interns = st.number_input("Internships", 0, 10, 0)
comm = st.slider("Communication Skill (1-10)", 1, 10, 5)
tech = st.slider("Technical Skills (1-5)", 1, 5, 3)
mock = st.number_input("Mock Test Score", 0, 100, 50)
cert = st.number_input("Certifications", 0, 20, 0)
spec = st.selectbox("Specialization", [0, 1, 2, 3, 4, 5])  # TODO: replace with actual names
gender = st.radio("Gender", [0, 1])  # TODO: confirm encoding
tier = st.selectbox("College Tier", [0, 1, 2])

# Predict
if st.button("Predict Placement"):
    features = np.array([[gender, cgpa, spec, interns, comm, tech, mock, cert, tier]])
    result = model.predict(features)[0]
    st.success("🎉 Likely to be Placed!" if result == 1 else "⚠ Unlikely to be Placed.")
