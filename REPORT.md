# 📄 Project Report: Campus Placement Predictor

## 👤 Team Member
- Sai Kushal Ranga

---

## 📝 Problem Statement
Campus placements play a crucial role in shaping the career paths of students. However, predicting whether a student is likely to get placed is often uncertain, as it depends on multiple factors such as academic performance, technical skills, communication abilities, and prior internships.  
There is a need for a data-driven solution that can analyze these factors and provide insights into a student’s placement chances, helping institutions and students take proactive steps for improvement.

---

## 🎯 Objective
The objective of this project is to build a **machine learning-based web application** that predicts the placement status of students based on their academic and skill-based profiles.  
Key goals:
- Provide an easy-to-use **Streamlit web interface** for predictions.  
- Enable **data visualization** to understand important factors influencing placement.  
- Develop a predictive model that achieves high accuracy and reliability.  
- Demonstrate how AI/ML can support career readiness in higher education.

---

## 📊 Dataset Description
- File Used: **cleaned_placement_data.csv**  
- Source: Processed version of a fictional campus placement dataset.  
- Number of Records: ~ (fill actual rows after checking)  
- Features include:
  - Gender (Male/Female)  
  - CGPA (5.0 – 10.0)  
  - Specialization (0–5 categories)  
  - Internships (0–5)  
  - Communication Skills (1–10)  
  - Technical Skills (1–5)  
  - Mock Test Score (0–100)  
  - Certifications (0–10)  
  - College Tier (Tier 1, Tier 2, Tier 3)  
- Target Variable: **Placement Status** (0: Not Placed, 1: Placed)

---

## 🔬 Methodology
1. **Data Preprocessing**
   - Handled missing values and ensured clean numerical ranges.  
   - Encoded categorical variables (e.g., gender, specialization, college tier).  
   - Normalized numerical features for consistent model training.  

2. **Model Training**
   - Algorithm used: **Random Forest Classifier**  
   - Dataset split: 80% training, 20% testing.  
   - Model saved as a `.pkl` file for deployment.  

3. **Web Application**
   - Built using **Streamlit**.  
   - Accepts student details via interactive sliders and input fields.  
   - Provides real-time placement prediction.  
   - Includes dataset visualizations for insights.  

---

## 📈 Model Performance
- Classifier: **Random Forest**  
- Accuracy Achieved: **87%**  
- Strength: Handles both categorical and numerical features effectively.  
- Output: Binary classification → "Placed" or "Not Placed".  

---

## 🖥️ System Design
**User Flow**:  
1. User opens Streamlit web app.  
2. Inputs academic & skill parameters.  
3. Model predicts placement status.  
4. Visualization charts display dataset insights.  

---

## ✅ Results
- Successfully built an end-to-end ML pipeline (data → model → deployment).  
- Interactive app helps students and educators analyze placement chances.  
- Visualization highlights the importance of **CGPA, internships, and skills** in placements.

---

## ⚡ Challenges Faced
- Preprocessing categorical data (specializations, tiers).  
- Balancing accuracy and interpretability of the model.  
- Deploying the Streamlit app under time constraints.  

---

## 🚀 Future Scope
- Add **resume parser** to auto-extract student details.  
- Predict **specific company/role fitment** instead of binary placement.  
- Expand dataset to include **extracurriculars, projects, hackathons**.  
- Provide personalized recommendations for improving placement chances.  

---

## 🏁 Conclusion
This project demonstrates how machine learning can be applied in the education domain to support placement readiness. By integrating a Random Forest model with a Streamlit interface, the solution provides both predictive power and usability.  
The **Campus Placement Predictor** serves as a step toward data-driven career guidance for students.  

---
