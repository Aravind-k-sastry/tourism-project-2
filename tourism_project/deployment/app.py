import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="aravindshaz3/predictor-model-2", filename="best_predictor.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Likelihood of Purchase app")
st.write("The Likelihood of purchase App is aan internal tool mean to predict the likelihood of a cusotmer picking a proposed plan.")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")


# Collect user input
Age = st.number_input("Age", min_value=18, max_value=61, value=36)
CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=127, value=9)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=3)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=5, value=4)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=3, max_value=5, value=4)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=22, value=4)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=3, value=1)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=98678, value=17500)


TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation?", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("HGender?", ["Female", "Male"])
ProductPitched = st.selectbox("ProductPitched?", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("MaritalStatus?", ["Single", "Divorced", "Married", "Unmarried"])
Designation = st.selectbox("Designation?", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])


Passport_input = st.selectbox("Passport?", ["Yes", "No"])
OwnCar_input = st.selectbox("Owns a Car?", ["Yes", "No"])

# Convert 'Yes'/'No' to 1/0 for numerical features
Passport = 1 if Passport_input == "Yes" else 0
OwnCar = 1 if OwnCar_input == "Yes" else 0


numeric_features = [

    'Age',               # Customer's age
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',    # Customer’s estimated salary
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',            # Number of years the customer has been with the bank
    'Occupation',
    'Gender',    # Whether the customer is an active member (binary: 0 or 1)
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]




# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age':Age,               # Customer's age
    'CityTier':CityTier,
    'DurationOfPitch':DurationOfPitch,
    'NumberOfPersonVisiting':NumberOfPersonVisiting,    # Customer’s estimated salary
    'NumberOfFollowups':NumberOfFollowups,
    'PreferredPropertyStar':PreferredPropertyStar,
    'NumberOfTrips':NumberOfTrips,
    'Passport':Passport,
    'PitchSatisfactionScore':PitchSatisfactionScore,
    'OwnCar':OwnCar,
    'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
    'MonthlyIncome':MonthlyIncome,


    'TypeofContact':TypeofContact,            # Number of years the customer has been with the bank
    'Occupation':Occupation,
    'Gender':Gender,    # Whether the customer is an active member (binary: 0 or 1)
    'ProductPitched':ProductPitched,
    'MaritalStatus':MaritalStatus,
    'Designation':Designation

}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Buy" if prediction == 1 else "Not buy"
    st.write(f"Based on the information provided, the customer is likely to {result} the plan.")
