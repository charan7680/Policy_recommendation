import streamlit as st
import joblib
import numpy as np
import pandas as pd
import manual_backend

# Load the trained model
gb_model = joblib.load(r"/workspaces/Policy_recommendation/ML_model.joblib")

# Define your feature lists
input_features = [
    "Gender", "Policy_Premium", 
    "Health_Heart_Disease", "Health_Tuberculosis", "Health_Cancer", "Health_Kidney_Disease",
    "Health_Asthma", "Health_Hypertension", "Health_COVID-19", "Health_Diabetes",
    "Disease_Heart_Disease", "Disease_Cancer", "Disease_Tuberculosis", "Disease_Kidney_Disease",
    "Disease_COVID-19", "Disease_Hypertension", "Disease_Asthma", "Disease_Diabetes"
]

# Define a mapping of class indices to insurance company names
company_name = [
    'Wilson-Lane', 'Reed PLC', 'Fry Group', 'Meyer-Ferguson', 'Barton, Morris and Scott', 
    'Love LLC', 'Bullock-Wu', 'Brown, Huerta and Miller', 'Torres Ltd', 'Morales LLC',
    'Martinez-Johnson', 'Hawkins-Green', 'Adams-Harris', 'Brown, Payne and Simmons',
    'Foster, Hubbard and Bell', 'Page-Smith', 'Savage-Lucas', 'Olson, Lopez and Parker', 
    'Harris-Edwards', 'Davis Inc', 'Moore, Cochran and Taylor', 'Burns-Murphy', 'Hunter, Clark and Woods',
    'Monroe LLC', 'Martin-Allen', 'Martinez, Smith and Daniels', 'Smith, Lee and Fitzgerald',
    'Burke-Cook', 'Valentine, Mcknight and Lloyd', 'Carroll Group'
]

# Streamlit app setup
st.title("Insurance Policy Prediction App")
st.write("This app predicts the best insurance policy based on health conditions and policy premium.")

# Get user inputs
gender = st.selectbox("Gender", options=["Male", "Female"])

# Input for Policy Premium
policy_premium = st.number_input(
    label="Enter Policy Premium",
    min_value=0,          # Minimum value allowed
    max_value=1000000,    # Maximum value allowed
    value=1000,           # Default value
    step=100              # Step size for increment/decrement
)

# Input for health conditions
st.write("Select the patient's health conditions:")
health_conditions = {}
for feature in input_features[2:10]:  # Skip "Gender" and "Policy_Premium"
    health_conditions[feature] = st.selectbox(feature, options=[0, 1], format_func=lambda x: "Yes" if x else "No")

# Automatically set disease values based on health conditions
disease_values = {
    disease: health_conditions[health] 
    for health, disease in zip(input_features[2:10], input_features[10:])
}

# Prepare input data
input_data = [
    1 if gender == "Male" else 0,  # Gender encoding
    policy_premium                 # Policy_Premium
] + list(health_conditions.values()) + list(disease_values.values())

input_df = pd.DataFrame([input_data], columns=input_features)
print(input_df)

# Predict and display results
if st.button("Predict Policy"):
    prediction = gb_model.predict([input_data])[0]
    st.success(f"Predicted Policy: {company_name[prediction]}")

    # For probabilistic results
    probabilities = gb_model.predict_proba([input_data])[0]
    st.write("Top 5 Prediction Probabilities:")
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    for idx in top_5_indices:
        st.write(f"Insurance Company: {company_name[idx]}")
