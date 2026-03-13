import streamlit as st
import requests

# 1. Setup the Page
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓")
st.title("🎓 Student Dropout Risk Predictor")
st.write("Enter student details below to analyze dropout risk using AI.")

# 2. Securely get the API Key from Streamlit Secrets
# (We will set this up in Step 6)
HF_API_KEY = st.secrets["HF_API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# 3. User Input Fields
name = st.text_input("Student Name")
attendance = st.slider("Attendance Percentage (%)", 0, 100, 75)
math_score = st.slider("Latest Math Score (0-100)", 0, 100, 50)

# 4. Prediction Logic
if st.button("Predict Risk"):
    if name:
        # Create the 'Letter' (Prompt) for the AI
        prompt = f"""
        Analyze the following student data:
        Name: {name}
        Attendance: {attendance}%
        Math Score: {math_score}/100

        Based on this, classify the Dropout Risk as 'High', 'Medium', or 'Low'.
        Provide a 2-sentence recommendation for the teacher.
        """
        
        with st.spinner("The AI is thinking..."):
            # Send request to Hugging Face
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()[0]['generated_text']
                # Clean up the output to show only the AI's reasoning
                ai_answer = result.replace(prompt, "").strip()
                
                st.success("Analysis Complete!")
                st.subheader("AI Assessment:")
                st.write(ai_answer)
            else:
                st.error("Wait, the AI is busy. Please try again in a moment!")
    else:
        st.warning("Please enter a student name.")