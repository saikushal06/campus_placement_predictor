# 🗺️ Project Lifecycle & Roadmap

## 📌 Phase 0: Ideation & Research
- Identified the problem: predicting campus placements based on academic and skill profiles.  
- Researched existing approaches and datasets related to placement prediction.  
- Defined key input features (CGPA, internships, communication skills, certifications, etc.).  

---

## 📌 Phase 1: Data Collection & Preprocessing
- Used **cleaned_placement_data.csv** as the dataset.  
- Performed data cleaning: handled missing values and ensured proper value ranges.  
- Encoded categorical features (Gender, Specialization, College Tier).  
- Normalized/standardized numerical features for consistency.  

---

## 📌 Phase 2: Model Development
- Experimented with multiple ML algorithms (Logistic Regression, Decision Tree, Random Forest).  
- Selected **Random Forest Classifier** for best performance.  
- Split dataset (80% training, 20% testing).  
- Achieved **87% accuracy** on test data.  
- Saved trained model (`placement_model.pkl`).  

---

## 📌 Phase 3: Application Development
- Built a web interface using **Streamlit**.  
- Designed input fields & sliders for easy data entry.  
- Integrated trained model for real-time placement prediction.  
- Added visualizations for dataset insights.  

---

## 📌 Phase 4: Deployment
- Deployed the Streamlit app to the cloud.  
- Linked live demo in README.  
- Recorded demo video showcasing project workflow.  

---

## 📌 Phase 5: Future Enhancements
- Add resume parsing to auto-fill student details.  
- Extend predictions to company/role recommendations.  
- Include extracurricular activities, hackathons, and projects in dataset.  
- Improve UI/UX for better usability.  

---

✅ This roadmap outlines the complete lifecycle of the **Campus Placement Predictor** project — from ideation to deployment and future improvements.
