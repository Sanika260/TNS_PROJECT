import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd


class ManufacturingFeatures(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Temperature_Pressure_Ratio: float
    Total_Cycle_Time: float
    Efficiency_Score: float
    Machine_Utilization: float
    Hour: int
    Day_of_Week_Num: int
    Month: int
    Shift_Night: int
    Shift_Evening: int
    Machine_Type_Type_B: int
    Material_Grade_Premium: int
    Material_Grade_Standard: int
    Day_of_Week_Monday: int
    Day_of_Week_Saturday: int
    Day_of_Week_Sunday: int
    Day_of_Week_Thursday: int
    Day_of_Week_Tuesday: int
    Day_of_Week_Wednesday: int

# Load the model and scaler
loaded_model = joblib.load('linear_regression_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

def predict_parts_per_hour(features: ManufacturingFeatures):
    """
    Predicts 'Parts_Per_Hour' using the trained Linear Regression model.
    """
    # Convert the input features to a pandas DataFrame
    input_df = pd.DataFrame([features.model_dump()])

    # Select numerical columns for scaling
    numerical_cols = ['Injection_Temperature', 'Injection_Pressure', 'Cycle_Time', 'Cooling_Time',
                      'Material_Viscosity', 'Ambient_Temperature', 'Machine_Age', 'Operator_Experience',
                      'Maintenance_Hours', 'Temperature_Pressure_Ratio', 'Total_Cycle_Time',
                      'Efficiency_Score', 'Machine_Utilization', 'Hour', 'Day_of_Week_Num', 'Month']

    # Apply the loaded scaler to the numerical features
    input_df[numerical_cols] = loaded_scaler.transform(input_df[numerical_cols])

    # Reorder columns to match the training data columns
    # This requires getting the original column order from X_train or X
    # Assuming X_train has the correct order after preprocessing
    input_df = input_df[X_train.columns]


    # Make prediction
    prediction = loaded_model.predict(input_df)

    return prediction[0]

app = FastAPI()

@app.post("/predict")
def predict(features: ManufacturingFeatures):
    """
    FastAPI endpoint to predict 'Parts_Per_Hour'.
    """
    prediction = predict_parts_per_hour(features)
    return {"predicted_parts_per_hour": prediction}