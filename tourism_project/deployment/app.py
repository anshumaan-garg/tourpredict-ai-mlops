# ============================================================================
# GRADIO frontend -- this is the DEPLOYED app.
# It overwrites the Streamlit app.py above, because Hugging Face removed the free
# Streamlit SDK (see the note above the Streamlit cell). Gradio is free on HF
# Spaces, so this Gradio version is what gets pushed to the Space.
# ============================================================================
import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load the trained model from the Hugging Face model hub
model_path = hf_hub_download(
    repo_id="NineKnox/tourism-wellness-model",
    filename="best_tourism_wellness_model_v1.joblib",
)
model = joblib.load(model_path)


def predict_purchase(Age, TypeofContact, CityTier, DurationOfPitch, Occupation,
                     Gender, NumberOfPersonVisiting, NumberOfFollowups,
                     ProductPitched, PreferredPropertyStar, MaritalStatus,
                     NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar,
                     NumberOfChildrenVisiting, Designation, MonthlyIncome):
    # Build a single-row DataFrame with the exact training feature columns.
    # Numeric-coded fields are cast so they never arrive as strings.
    input_data = pd.DataFrame([{
        "Age": float(Age),
        "TypeofContact": TypeofContact,
        "CityTier": int(CityTier),
        "DurationOfPitch": float(DurationOfPitch),
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": float(NumberOfPersonVisiting),
        "NumberOfFollowups": float(NumberOfFollowups),
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": int(PreferredPropertyStar),
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": float(NumberOfTrips),
        "Passport": int(Passport),
        "PitchSatisfactionScore": float(PitchSatisfactionScore),
        "OwnCar": int(OwnCar),
        "NumberOfChildrenVisiting": float(NumberOfChildrenVisiting),
        "Designation": Designation,
        "MonthlyIncome": float(MonthlyIncome),
    }])
    pred = int(model.predict(input_data)[0])
    proba = float(model.predict_proba(input_data)[0][1])
    if pred == 1:
        return f"Likely to PURCHASE the Wellness Package (probability: {proba:.2%})"
    return f"Unlikely to purchase the Wellness Package (probability: {proba:.2%})"


demo = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Number(label="Age", value=35),
        gr.Dropdown(["Self Enquiry", "Company Invited"], label="Type of Contact", value="Self Enquiry"),
        gr.Dropdown([1, 2, 3], label="City Tier", value=1),
        gr.Number(label="Duration of Pitch (minutes)", value=15),
        gr.Dropdown(["Salaried", "Small Business", "Large Business", "Free Lancer"], label="Occupation", value="Salaried"),
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Number(label="Number of Persons Visiting", value=3),
        gr.Number(label="Number of Follow-ups", value=3),
        gr.Dropdown(["Basic", "Deluxe", "Standard", "Super Deluxe", "King"], label="Product Pitched", value="Deluxe"),
        gr.Dropdown([3, 4, 5], label="Preferred Property Star", value=3),
        gr.Dropdown(["Married", "Divorced", "Single"], label="Marital Status", value="Married"),
        gr.Number(label="Number of Trips per Year", value=2),
        gr.Dropdown([0, 1], label="Holds Passport (0=No, 1=Yes)", value=1),
        gr.Slider(1, 5, step=1, label="Pitch Satisfaction Score", value=3),
        gr.Dropdown([0, 1], label="Owns Car (0=No, 1=Yes)", value=1),
        gr.Number(label="Number of Children Visiting (below 5)", value=0),
        gr.Dropdown(["Executive", "Manager", "Senior Manager", "AVP", "VP"], label="Designation", value="Manager"),
        gr.Number(label="Monthly Income", value=20000),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Wellness Tourism Package - Purchase Prediction",
    description="Enter the customer profile and interaction details to predict whether the customer will purchase the Wellness Tourism Package.",
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
