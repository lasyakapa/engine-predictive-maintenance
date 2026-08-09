
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# Page configuration
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="🔧"
)


# Load model from Hugging Face Model Hub
@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="Lasya679/engine-predictive-maintenance-model",
        filename="engine_predictive_maintenance_model.joblib",
        repo_type="model"
    )

    return joblib.load(model_path)


model = load_model()


# Application title
st.title("Engine Predictive Maintenance")

st.write(
    "Enter the engine sensor measurements below "
    "to predict the engine condition."
)


# User inputs
engine_rpm = st.number_input(
    "Engine RPM",
    min_value=0.0,
    value=800.0
)

lub_oil_pressure = st.number_input(
    "Lub Oil Pressure",
    min_value=0.0,
    value=3.0
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0,
    value=6.0
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    min_value=0.0,
    value=2.0
)

lub_oil_temp = st.number_input(
    "Lub Oil Temperature",
    min_value=0.0,
    value=76.0
)

coolant_temp = st.number_input(
    "Coolant Temperature",
    min_value=0.0,
    value=80.0
)


# Create DataFrame from user inputs
input_data = pd.DataFrame(
    {
        "Engine rpm": [engine_rpm],
        "Lub oil pressure": [lub_oil_pressure],
        "Fuel pressure": [fuel_pressure],
        "Coolant pressure": [coolant_pressure],
        "lub oil temp": [lub_oil_temp],
        "Coolant temp": [coolant_temp],
    }
)


# Display input data
st.subheader("Input Sensor Values")
st.dataframe(input_data)


# Generate prediction
if st.button("Predict Engine Condition"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("Engine Condition: 1")
    else:
        st.warning("Engine Condition: 0")
