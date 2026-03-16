import streamlit as st
import joblib
import numpy as np

# 1. Load the "Brain" we created in Jupyter
# We use st.cache_resource so the app only loads the file once (saves memory)
@st.cache_resource
def load_my_model():
    return joblib.load("student_dropout_model.pkl")

model = load_my_model()

# 2. Setup the Page Title and Styling
st.set_page_config(page_title="3MTT Dropout Predictor", page_icon="🎓")
st.title("🎓 Student Dropout Risk Predictor")
st.write("Enter student data below to get an instant risk assessment.")

# 3. Create the Input Fields (Matching your dataset features)
# Note: We use the exact names we used in Jupyter: attendance and test scores
st.subheader("Student Stats")
name = st.text_input("Full Name of Student")

# The numbers below match the ranges in your Nigerian dataset
attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
test_score = st.slider("Average Test Score (0-100)", 0, 100, 50)

# 4. The Prediction Logic
if st.button("Analyze Student Risk"):
    if name:
        # Step A: Prepare the data for the AI
        # The AI expects a 2D list: [[attendance, score]]
        features = np.array([[attendance, test_score]])
        
        # Step B: Make the prediction (0 = Stay, 1 = Dropout)
        prediction = model.predict(features)[0]
        
        # Step C: Show the result to the teacher
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **High Risk:** {name} is at risk of dropping out.")
            st.markdown("""
            **Recommended Actions:**
            * Reach out to parents/guardians.
            * Provide additional academic support in Math/English.
            * Check for external factors (distance to school, fees).
            """)
        else:
            st.success(f"✅ **Low Risk:** {name} is likely to continue their education.")
            st.write("Keep up the great work and continue regular monitoring!")
    else:
        st.warning("Please enter a student name to continue.")

# 5. Add a Footer for your 3MTT Project
st.sidebar.info("Built for the 3MTT NextGen AI/ML Track.")
st.sidebar.markdown("---")
st.sidebar.write("Model: Random Forest Classifier")
st.sidebar.write("Dataset: Nigeria Student Dropout Data")