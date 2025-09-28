# predict.py
import requests

# FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Example input (already one-hot encoded based on your Pydantic schema)
data = {
    "Injection_Temperature": 221,
    "Injection_Pressure": 136,
    "Cycle_Time": 28.7,
    "Cooling_Time": 13.6,
    "Material_Viscosity": 375.5,
    "Ambient_Temperature": 28,
    "Machine_Age": 3.8,
    "Operator_Experience": 11.2,
    "Maintenance_Hours": 64,
    "Temperature_Pressure_Ratio": 1.625,
    "Total_Cycle_Time": 42.3,
    "Efficiency_Score": 0.063,
    "Machine_Utilization": 0.51,
    "Hour": 0,
    "Day_of_Week_Num": 4,
    "Month": 1,

    "Shift_Night": 0,
    "Shift_Evening": 1,
    "Machine_Type_Type_B": 1,
    "Material_Grade_Premium": 0,
    "Material_Grade_Standard": 0,
    "Day_of_Week_Monday": 0,
    "Day_of_Week_Saturday": 0,
    "Day_of_Week_Sunday": 0,
    "Day_of_Week_Thursday": 1,
    "Day_of_Week_Tuesday": 0,
    "Day_of_Week_Wednesday": 0
}

# Send POST request
response = requests.post(url, json=data)

# Print response
if response.status_code == 200:
    print("✅ Prediction:", response.json())
else:
    print("❌ Error:", response.status_code, response.text)
