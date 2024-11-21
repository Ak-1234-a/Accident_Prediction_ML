import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('accident_model.pkl')

# Title of the app
st.title("Accident Severity Prediction")

# Collect user input
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
light_conditions = st.selectbox("Light Conditions", ["Daylight", "Darkness"])
sex_of_driver = st.selectbox("Sex of Driver", ["Male", "Female"])
vehicle_type = st.slider("Vehicle Type (encoded)", 0, 10, 1)
speed_limit = st.number_input("Speed Limit (km/h)", min_value=0, max_value=120, value=40, step=5)
pedestrian_crossing = st.slider("Pedestrian Crossing (encoded)", 0, 5, 0)
road_type = st.slider("Road Type (encoded)", 0, 6, 0)
special_conditions = st.slider("Special Conditions at Site (encoded)", 0, 2, 0)
num_passengers = st.slider("Number of Passengers", 0, 10, 1)

# Map inputs to numeric values (if necessary)
day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
light_mapping = {"Daylight": 0, "Darkness": 1}
sex_mapping = {"Male": 0, "Female": 1}

# Convert user inputs to numeric
input_data = np.array([
    day_mapping[day_of_week],
    light_mapping[light_conditions],
    sex_mapping[sex_of_driver],
    vehicle_type,
    speed_limit,
    pedestrian_crossing,
    road_type,
    special_conditions,
    num_passengers
]).reshape(1, -1)

# Predict severity
if st.button("Predict"):
    prediction = model.predict(input_data)
    severity_mapping = {0: "Slight", 1: "Serious", 2: "Fatal"}
    severity = severity_mapping[prediction[0]]
    st.write(f"Predicted Accident Severity: **{severity}**")
