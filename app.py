import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("models/placement_model.pkl", "rb"))

st.title("Campus Placement Predictor")

# Inputs
cgpa = st.slider("CGPA", 5.0, 10.0)
interns = st.number_input("Internships", 0, 5)
comm = st.slider("Communication Skill", 1, 10)
tech = st.slider("Technical Skills", 1, 5)
mock = st.number_input("Mock Test Score", 0, 100)
cert = st.number_input("Certifications", 0, 10)
spec = st.selectbox("Specialization", [0, 1, 2, 3, 4, 5]) # Replace with encoding
gender = st.radio("Gender", [0, 1]) # M=1, F=0
tier = st.selectbox("College Tier", [0, 1, 2]) # Encode as in training

# Predict
if st.button("Predict Placement"):
    features = np.array([[gender, cgpa, spec, interns, comm, tech, mock, cert, tier]])
    result = model.predict(features)[0]
    st.success("🎉 Likely to be Placed!" if result == 1 else "⚠ Unlikely to be Placed.")