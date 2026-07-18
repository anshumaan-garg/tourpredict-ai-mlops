import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model from the Hugging Face model hub
model_path = hf_hub_download(
    repo_id="YOUR_HF_USERNAME/tourism-wellness-model",
    filename="best_tourism_wellness_model_v1.joblib",
)
model = joblib.load(model_path)

# ------------------------------ Streamlit UI --------------------------------
st.title("Wellness Tourism Package - Purchase Prediction")
st.write(
    "This app predicts whether a customer is likely to purchase the "
    "**Wellness Tourism Package** based on their profile and interaction data. "
    "Fill in the customer details below and click **Predict**."
)

# Categorical inputs
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Married", "Divorced", "Single"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Numeric / ordinal inputs
city_tier = st.selectbox("City Tier", [1, 2, 3])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
passport = st.selectbox("Holds Passport", [0, 1])
own_car = st.selectbox("Owns Car", [0, 1])

age = st.number_input("Age", min_value=18, max_value=100, value=35)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=200, value=15)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=2)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
number_of_children_visiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=20000, step=500)

# Assemble the inputs into a single-row DataFrame (columns must match training)
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Likely to PURCHASE the Wellness Package (probability: {proba:.2%})")
    else:
        st.info(f"Unlikely to purchase the Wellness Package (probability: {proba:.2%})")
