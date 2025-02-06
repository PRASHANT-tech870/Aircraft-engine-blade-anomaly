import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import gdown
import requests
from sklearn.ensemble import RandomForestClassifier

# Function to download the file from Google Drive using gdown
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# File ID from Google Drive URL
file_id = "1WOL9TQLPZ-RRon8vc1sDtY7IKuYH0ydt"  # Extracted from the original URL
model_file = "rf_model_cpu.pkl"

# Check if the file already exists, if not, download it
if not os.path.exists(model_file):
    try:
        download_file_from_google_drive(file_id, model_file)
        st.success("Model file downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model file: {e}")

# Load the trained model after downloading it
try:
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Real sensor names and corresponding model feature names
sensor_names = {
    "Cycle": "Cycle", 
    "OpSet1": "OpSet1", "OpSet2": "OpSet2", "OpSet3": "OpSet3",
    "Primary Temperature Reading": "SensorMeasure1", 
    "Secondary Temperature Reading": "SensorMeasure2", 
    "Tertiary Temperature Reading": "SensorMeasure3", 
    "Quaternary Temperature Reading": "SensorMeasure4", 
    "Primary Pressure Reading": "SensorMeasure5", 
    "Secondary Pressure Reading": "SensorMeasure6", 
    "Tertiary Pressure Reading": "SensorMeasure7", 
    "Quaternary Pressure Reading": "SensorMeasure8", 
    "Primary Speed Reading": "SensorMeasure9", 
    "Secondary Speed Reading": "SensorMeasure10", 
    "Tertiary Speed Reading": "SensorMeasure11", 
    "Quaternary Speed Reading": "SensorMeasure12", 
    "Primary Vibration Reading": "SensorMeasure13", 
    "Secondary Vibration Reading": "SensorMeasure14", 
    "Primary Flow Reading": "SensorMeasure15", 
    "Secondary Flow Reading": "SensorMeasure16", 
    "Tertiary Flow Reading": "SensorMeasure17", 
    "Pressure Ratio": "SensorMeasure18", 
    "Efficiency Indicator": "SensorMeasure19", 
    "Power Setting": "SensorMeasure20", 
    "Fuel Flow Rate": "SensorMeasure21"
}

# Predefined values for GOOD, MODERATE, VERY BAD
predefined_values = {
    "GOOD": [64,20.0004,0.7007,100.0,491.19,606.79,1477.26,1234.25,9.35,13.61,332.51,2323.71,8709.48,1.07,43.86,313.57,2387.77,8050.58,9.1851,0.02,364,2324,100.0,24.6,14.6684],
    "MODERATE": [213,10.0018,0.25,100.0,489.05,604.4,1492.63,1306.34,10.52,15.47,397.07,2318.98,8778.54,1.26,45.37,373.56,2388.16,8141.38,8.571,0.03,369,2319,100.0,28.74,17.2585],
    "VERY BAD": [263,10.0077,0.2501,100.0,489.05,604.86,1507.7,1318.06,10.52,15.47,401.91,2319.43,8816.35,1.27,45.7,379.16,2388.61,8170.26,8.4897,0.03,372,2319,100.0,28.85,17.3519]
}

# FastAPI endpoint to send data to
FASTAPI_URL = "http://127.0.0.1:8000/update_sensor_data/"

# Function to send sensor data to FastAPI backend
def send_data_to_fastapi(sensor_data, prediction):
    # Add the prediction to the sensor data
    sensor_data["prediction"] = prediction
    
    # Ensure the sensor data has the correct field names that match the FastAPI model
    sensor_data = {
        "cycle": sensor_data["cycle"],
        
        # Operating Settings
        "operating_setting_1": sensor_data["operating_setting_1"],
        "operating_setting_2": sensor_data["operating_setting_2"],
        "operating_setting_3": sensor_data["operating_setting_3"],
        
        # Temperature Sensors
        "primary_temperature": sensor_data["primary_temperature"],
        "secondary_temperature": sensor_data["secondary_temperature"],
        "tertiary_temperature": sensor_data["tertiary_temperature"],
        "quaternary_temperature": sensor_data["quaternary_temperature"],
        
        # Pressure Sensors
        "primary_pressure": sensor_data["primary_pressure"],
        "secondary_pressure": sensor_data["secondary_pressure"],
        "tertiary_pressure": sensor_data["tertiary_pressure"],
        "quaternary_pressure": sensor_data["quaternary_pressure"],
        
        # Speed/Rotation Sensors
        "primary_speed": sensor_data["primary_speed"],
        "secondary_speed": sensor_data["secondary_speed"],
        "tertiary_speed": sensor_data["tertiary_speed"],
        "quaternary_speed": sensor_data["quaternary_speed"],
        
        # Vibration/Mechanical Sensors
        "primary_vibration": sensor_data["primary_vibration"],
        "secondary_vibration": sensor_data["secondary_vibration"],
        
        # Flow Sensors
        "primary_flow": sensor_data["primary_flow"],
        "secondary_flow": sensor_data["secondary_flow"],
        "tertiary_flow": sensor_data["tertiary_flow"],
        
        # Additional Sensors
        "pressure_ratio": sensor_data["pressure_ratio"],
        "efficiency_indicator": sensor_data["efficiency_indicator"],
        "power_setting": sensor_data["power_setting"],
        "fuel_flow_rate": sensor_data["fuel_flow_rate"],
        
        # Prediction
        "prediction": sensor_data["prediction"]
    }

    # Send POST request to FastAPI
    try:
        response = requests.post(FASTAPI_URL, json=sensor_data)
        st.write(response.json())  # This will show any error details returned by FastAPI

        if response.status_code == 200:
            st.success("Data sent successfully to the server!")
        else:
            st.error(f"Failed to send data to the server. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error sending data to FastAPI: {e}")


# Streamlit UI
st.title("Engine Health Predictor")

# Autofill buttons
for label in predefined_values:
    if st.button(f"Autofill for {label}"):
        for real_name, model_name in sensor_names.items():
            st.session_state[model_name] = predefined_values[label][list(sensor_names.keys()).index(real_name)]

# Text input for pasting comma-separated values
user_input = st.text_area("Paste comma-separated values", "")

# Apply button for user input
if st.button("Apply"):
    if user_input:
        input_values = list(map(float, user_input.split(',')))
        
        if len(input_values) == len(sensor_names):
            for real_name, model_name in sensor_names.items():
                st.session_state[model_name] = input_values[list(sensor_names.keys()).index(real_name)]
        else:
            st.error("Invalid input. Ensure the number of values matches the required fields.")

# User input fields with real sensor names displayed
input_data = []
for real_name, model_name in sensor_names.items():
    value = st.number_input(real_name, key=model_name, value=st.session_state.get(model_name, 0.0))
    input_data.append(value)

# Convert input_data to a pandas DataFrame with dtype float32
input_df = pd.DataFrame([input_data], columns=list(sensor_names.values()))
input_df = input_df.astype('float32')

# Submit button
if st.button("Submit"):
    prediction = model.predict(input_df)[0]  # Pass DataFrame to model
    result_map = {0: "GOOD", 1: "MODERATE", 2: "VERY BAD"}
    prediction_result = result_map.get(prediction, 'Unknown')
    print(prediction_result, type(prediction_result))
    st.success(f"Predicted Condition: {prediction_result}")
    
    # Prepare the sensor data for sending
    sensor_data = {
    "cycle": input_data[0],  # Engine operational cycle number
    
    # Operating Settings
    "operating_setting_1": input_data[1],  # First operational parameter (OpSet1)
    "operating_setting_2": input_data[2],  # Second operational parameter (OpSet2)
    "operating_setting_3": input_data[3],  # Third operational parameter (OpSet3)
    
    # Temperature Sensors
    "primary_temperature": input_data[4],    # SensorMeasure1
    "secondary_temperature": input_data[5],  # SensorMeasure2
    "tertiary_temperature": input_data[6],   # SensorMeasure3
    "quaternary_temperature": input_data[7], # SensorMeasure4
    
    # Pressure Sensors
    "primary_pressure": input_data[8],    # SensorMeasure5
    "secondary_pressure": input_data[9],  # SensorMeasure6
    "tertiary_pressure": input_data[10],  # SensorMeasure7
    "quaternary_pressure": input_data[11],# SensorMeasure8
    
    # Speed/Rotation Sensors
    "primary_speed": input_data[12],    # SensorMeasure9
    "secondary_speed": input_data[13],  # SensorMeasure10
    "tertiary_speed": input_data[14],   # SensorMeasure11
    "quaternary_speed": input_data[15], # SensorMeasure12
    
    # Vibration/Mechanical Sensors
    "primary_vibration": input_data[16],   # SensorMeasure13
    "secondary_vibration": input_data[17], # SensorMeasure14
    
    # Flow Sensors
    "primary_flow": input_data[18],    # SensorMeasure15
    "secondary_flow": input_data[19],  # SensorMeasure16
    "tertiary_flow": input_data[20],   # SensorMeasure17
    
    # Additional Sensors
    "pressure_ratio": input_data[21],      # SensorMeasure18
    "efficiency_indicator": input_data[22],# SensorMeasure19
    "power_setting": input_data[23],       # SensorMeasure20
    "fuel_flow_rate": input_data[24]       # SensorMeasure21
    }

    
    # Send data to FastAPI
    send_data_to_fastapi(sensor_data, prediction_result)
