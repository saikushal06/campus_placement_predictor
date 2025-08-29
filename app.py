import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("models/placement_model.pkl", "rb"))

st.set_page_config(page_title="Campus Placement Predictor", layout="centered")

st.title("🎓 Campus Placement Predictor")
st.write("Predict your placement chances based on your academic and skill profile.")

# Collect user input features
gender = st.selectbox("Gender", ("Male", "Female"))
cgpa = st.slider("CGPA", 5.0, 10.0, 7.0)
specialization = st.selectbox("Specialization", ["0", "1", "2", "3", "4", "5"])
internships = st.slider("Number of Internships", 0, 5, 1)
comm_skill = st.slider("Communication Skills (1-10)", 1, 10, 5)
tech_skill = st.slider("Technical Skills (1-5)", 1, 5, 3)
mock_test = st.slider("Mock Test Score (0-100)", 0, 100, 50)
certifications = st.slider("Certifications Completed", 0, 10, 2)
college_tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])

# Encode categorical variables
gender_val = 0 if gender == "Male" else 1
specialization_val = int(specialization)
college_tier_val = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}[college_tier]

# Prepare input array
input_data = np.array([[gender_val, cgpa, specialization_val, internships,
                        comm_skill, tech_skill, mock_test, certifications,
                        college_tier_val]])

# Prediction button
if st.button("Predict Placement"):
    prob = model.predict_proba(input_data)[0]  # probability for both classes
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Results")
    st.write(f"**Probability of Getting Placed:** {prob[1]*100:.2f}%")
    st.write(f"**Probability of Not Getting Placed:** {prob[0]*100:.2f}%")

    if prediction == 1:
        st.success("🎉 Likely to be Placed")
    else:
        st.error("❌ Not Likely to be Placed")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
