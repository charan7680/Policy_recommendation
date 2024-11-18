import joblib
import fitz  # PyMuPDF
import pandas as pd
import numpy as np

# Load the trained model
gb_model = joblib.load(r"/workspaces/Policy_recommendation/ML_model.joblib")

# Define input features
input_features = [
    "Gender", "Policy_Premium", 
    "Health_Heart_Disease", "Health_Tuberculosis", "Health_Cancer",
    "Health_Kidney_Disease", "Health_Asthma", "Health_Hypertension", 
    "Health_COVID-19", "Health_Diabetes",
    "Disease_Heart_Disease", "Disease_Cancer", "Disease_Tuberculosis", 
    "Disease_Kidney_Disease", "Disease_COVID-19", "Disease_Hypertension", 
    "Disease_Asthma", "Disease_Diabetes"
]

# Define company names
company_name = [
    'Wilson-Lane', 'Reed PLC', 'Fry Group', 'Meyer-Ferguson', 'Barton, Morris and Scott', 
    'Love LLC', 'Bullock-Wu', 'Brown, Huerta and Miller', 'Torres Ltd', 'Morales LLC',
    'Martinez-Johnson', 'Hawkins-Green', 'Adams-Harris', 'Brown, Payne and Simmons',
    'Foster, Hubbard and Bell', 'Page-Smith', 'Savage-Lucas', 'Olson, Lopez and Parker', 
    'Harris-Edwards', 'Davis Inc', 'Moore, Cochran and Taylor', 'Burns-Murphy', 
    'Hunter, Clark and Woods', 'Monroe LLC', 'Martin-Allen', 'Martinez, Smith and Daniels', 
    'Smith, Lee and Fitzgerald', 'Burke-Cook', 'Valentine, Mcknight and Lloyd', 'Carroll Group'
]

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open("pdf", pdf_file.read()) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to parse extracted text and fill input features
def parse_pdf_text(text):
    """
    Parses the extracted text from the PDF to identify the health features.
    Ensures that 'Detected' corresponds to 1 and 'Not Detected' corresponds to 0.
    """
    # Initialize health feature dictionary with default values as 0 (not detected)
    health_data = {
        "Gender": 0,  # Default; assume manual entry for gender in frontend
        "Policy_Premium": 0,  # Default; assume manual entry for Policy_Premium
        "Health_Heart_Disease": 0,
        "Health_Tuberculosis": 0,
        "Health_Cancer": 0,
        "Health_Kidney_Disease": 0,
        "Health_Asthma": 0,
        "Health_Hypertension": 0,
        "Health_COVID-19": 0,
        "Health_Diabetes": 0
    }

    # Extract and match conditions from text
    for condition in health_data.keys():
        if condition.startswith("Health_"):
            if f"{condition.split('_')[1]}: Detected" in text:
                health_data[condition] = 1
            elif f"{condition.split('_')[1]}: Not Detected" in text:
                health_data[condition] = 0

    return health_data

# Function to prepare final input by filling disease indicators based on health conditions
def prepare_model_input(health_data, policy_premium, gender):
    # Update Policy_Premium and Gender in health data
    health_data["Policy_Premium"] = policy_premium
    health_data["Gender"] = 1 if gender == "Male" else 0

    # Automatically fill disease features based on health data
    disease_features = {
        "Disease_Heart_Disease": health_data["Health_Heart_Disease"],
        "Disease_Cancer": health_data["Health_Cancer"],
        "Disease_Tuberculosis": health_data["Health_Tuberculosis"],
        "Disease_Kidney_Disease": health_data["Health_Kidney_Disease"],
        "Disease_COVID-19": health_data["Health_COVID-19"],
        "Disease_Hypertension": health_data["Health_Hypertension"],
        "Disease_Asthma": health_data["Health_Asthma"],
        "Disease_Diabetes": health_data["Health_Diabetes"]
    }

    # Combine health and disease features for model input
    return {**health_data, **disease_features}

# Function to predict policy and probabilities
def predict_policy(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=input_features)

    # Predict the policy
    prediction = gb_model.predict(input_df)[0]

    # Predict probabilities
    probabilities = gb_model.predict_proba(input_df)[0]

    return prediction, probabilities

# Function to get top 5 predictions
def get_top_predictions(probabilities):
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    return [(company_name[idx], probabilities[idx]) for idx in top_5_indices]
