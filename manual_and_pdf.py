import streamlit as st
import joblib
import numpy as np
import pandas as pd
import ast
from PyPDF2 import PdfReader
import google.generativeai as genai

# Load the trained model
model = joblib.load(r"ML_model.joblib")

# Define input features
input_features = ["Gender","Policy_Premium","Health_Hypertension","Health_Cancer","Health_Diabetes","Health_COVID-19","Health_Heart_Disease","Health_Kidney_Disease","Health_Asthma","Health_Tuberculosis","Disease_Cancer","Disease_Hypertension","Disease_Diabetes","Disease_COVID-19","Disease_Heart_Disease","Disease_Kidney_Disease","Disease_Asthma","Disease_Tuberculosis"]

# Define a mapping of class indices to insurance company names
company_name = ['Adams-Harris', 'Barton, Morris and Scott', 'Brown, Huerta and Miller', 'Brown, Payne and Simmons', 'Bullock-Wu', 'Burke-Cook', 'Burns-Murphy', 'Carroll Group', 'Davis Inc', 'Foster, Hubbard and Bell', 'Fry Group', 'Harris-Edwards', 'Hawkins-Green', 'Hunter, Clark and Woods', 'Love LLC', 'Martin-Allen', 'Martinez, Smith and Daniels', 'Martinez-Johnson', 'Meyer-Ferguson', 'Monroe LLC', 'Moore, Cochran and Taylor', 'Morales LLC', 'Olson, Lopez and Parker', 'Page-Smith', 'Reed PLC', 'Savage-Lucas', 'Smith, Lee and Fitzgerald', 'Torres Ltd', 'Valentine, Mcknight and Lloyd', 'Wilson-Lane']


# Streamlit app setup
st.title("Insurance Policy Prediction App")
st.write("This app predicts the best insurance policy based on health conditions and policy premium.")

# Select input method
input_method = st.radio("Choose Input Method:", options=["Upload Health Report (PDF)", "Manual Entry"])

if input_method == "Upload Health Report (PDF)":
    uploaded_file = st.file_uploader("Upload Health Report PDF", type="pdf")

    if uploaded_file is not None:
        # Process the file using Gemini API
        genai.configure(api_key=[api_key])
        Gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """\nConsider the medical report and extract the medical records of the patient in this python dictionary format

        "json
        {
            "Gender": 1 if male 0 if female else 2, 
            "Health_Heart_Disease": 1 if true  else 0,
            "Health_Tuberculosis": 1 if true  else 0,
            "Health_Cancer": 1 if true  else 0,
            "Health_Kidney_Disease": 1 if true  else 0,
            "Health_Asthma": 1 if true  else 0,
            "Health_Hypertension": 1 if true  else 0,
            "Health_COVID-19": 1 if true  else 0,
            "Health_Diabetes": 1 if true  else 0
        }"

        DONT INCLUDE ANY OTHER TEXT. ONLY give me the dictionary"""

        pdf_reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        response = Gemini_model.generate_content(pdf_text + prompt)
        print(response.text)
        extracted_data = ast.literal_eval(response.text.split("json")[1].replace("```", ""))
        
        # Prepare input data
        extracted_data["Policy_Premium"] = st.number_input(
            label="Enter Policy Premium",
            min_value=0,
            max_value=15000,
            value=1000,
            step=100
        )
        extracted_data.update({
            f"Disease_{key.split('Health_')[1]}": value
            for key, value in extracted_data.items() if key.startswith("Health_")
        })
        print(extracted_data)
        input_data = [extracted_data[feature] for feature in input_features]
        print(input_data)

elif input_method == "Manual Entry":
    # Manual inputs
    gender = st.selectbox("Gender", options=["Male", "Female","Others"])
    policy_premium = st.number_input(
        label="Enter Policy Premium",
        min_value=0,
        max_value=15000,
        value=1000,
        step=100
    )
    
    st.write("Select the patient's health conditions:")
    health_conditions = {}
    for feature in input_features[2:10]:  # Skip "Gender" and "Policy_Premium"
        health_conditions[feature] = st.selectbox(feature, options=[0, 1], format_func=lambda x: "Yes" if x else "No")

    # Automatically set disease values
    disease_values = {
        f"Disease_{key.split('_')[1]}": value
        for key, value in health_conditions.items()
    }

    # Prepare input data
    input_data = [
        1 if gender == "Male" else 0 if gender=="Female"  else 2 ,  # Gender encoding
        policy_premium
    ] + list(health_conditions.values()) + list(disease_values.values())

# Prediction button
if st.button("Predict Policy"):
    prediction = model.predict([input_data])[0]
    st.success(f"Predicted Policy: {company_name[prediction]}")

    # Probabilistic results
    probabilities = model.predict_proba([input_data])[0]
    st.write("Top 5 Prediction Probabilities:")
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    for idx in top_5_indices:
        st.write(f"Insurance Company: {company_name[idx]}")
