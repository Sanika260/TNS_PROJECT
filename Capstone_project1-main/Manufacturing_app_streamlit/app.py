import streamlit as st
import joblib
import pandas as pd

# Custom CSS for light blue background, white inputs, and black text
st.markdown(
    """
    <style>
    /* Page background */
    .reportview-container, .main {
        background-color: azure;
    }
    /* Header text color */
    .main-header {
        color: black;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Subheader text color */
    .sub-header {
        color: black;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    /* Black labels for selectbox */
    .stSelectbox>div>div>div>label {
        color: black !important;
        font-weight: bold;
        font-size: 18px;
    }
    /* Number input text field white background and black text */
    .stNumberInput>div>div>input {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
        border: 2px solid black;
        padding: 8px;
        font-size: 16px;
    }
    /* Help text in black */
    .helptext {
        font-style: italic;
        color: black;
        font-size: 14px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #FFD43B;
        color: black;
        font-weight: bold;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: none;
        margin-top: 20px;
    }
    /* Output text in black */
    .output-text {
        color: black;
        font-size: 20px;
        font-weight: 600;
        margin-top: 20px;
    }
    /* Footer */
    .footer {
        font-size: 14px;
        color: black;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown('<div class="main-header">Manufacturing Parts Prediction</div>', unsafe_allow_html=True)

    # Load model, scaler, columns, RMSE
    try:
        model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        rmse = joblib.load('rmse.pkl')
        st.success("Model, scaler, and columns loaded successfully!")
    except FileNotFoundError:
        st.error("Error loading model files. Please ensure all model files are in the correct directory.")
        return

    st.markdown('<div class="sub-header">Enter Manufacturing Parameters:</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    def number_input_with_help(label, min_val, max_val, default, help_text, container):
        val = container.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
        )
        container.markdown(f'<div class="helptext">{help_text}</div>', unsafe_allow_html=True)
        return val

    with col1:
        injection_temp = number_input_with_help("Injection_Temperature", 100.0, 300.0, 221.0, "Example: 221 (range 100-300)", col1)
        cycle_time = number_input_with_help("Cycle_Time (seconds)", 15.0, 60.0, 28.7, "Example: 28.7 (range 15-60s)", col1)
        material_viscosity = number_input_with_help("Material_Viscosity", 50.0, 500.0, 375.5, "Example: 375.5 (range 50-500)", col1)
        machine_age = number_input_with_help("Machine_Age (years)", 0.0, 20.0, 3.8, "Example: 3.8 (range 0-20 years)", col1)
        temperature_pressure_ratio = number_input_with_help("Temperature_Pressure_Ratio", 0.5, 3.0, 1.625, "Example: 1.625 (range 0.5-3.0)", col1)
        efficiency_score = number_input_with_help("Efficiency_Score", 0.01, 1.0, 0.063, "Example: 0.063 (range 0.01-1.0)", col1)

    with col2:
        injection_pressure = number_input_with_help("Injection_Pressure", 50.0, 200.0, 136.0, "Example: 136 (range 50-200)", col2)
        cooling_time = number_input_with_help("Cooling_Time (seconds)", 5.0, 30.0, 13.6, "Example: 13.6 (range 5-30s)", col2)
        ambient_temp = number_input_with_help("Ambient_Temperature (°C)", 10.0, 40.0, 28.0, "Example: 28 (range 10-40 °C)", col2)
        operator_exp = number_input_with_help("Operator_Experience (years)", 0.0, 30.0, 11.2, "Example: 11.2 (range 0-30 years)", col2)
        total_cycle_time = number_input_with_help("Total_Cycle_Time (seconds)", 15.0, 80.0, 42.3, "Example: 42.3 (range 15-80s)", col2)
        machine_utilization = number_input_with_help("Machine_Utilization", 0.0, 1.0, 0.51, "Example: 0.51 (range 0.0-1.0)", col2)

    maintenance_hours = st.number_input(
        "Maintenance_Hours (hours/month)",
        min_value=0.0,
        max_value=100.0,
        value=64.0,
        help="Example: 64 (range 0-100 hrs/month)"
    )

    shift = st.selectbox("Shift", ['Day', 'Evening', 'Night'])
    machine_type = st.selectbox("Machine_Type", ['Type_A', 'Type_B', 'Type_C'])
    material_grade = st.selectbox("Material_Grade", ['Economy', 'Premium', 'Standard'])
    day_of_week = st.selectbox("Day_of_Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    if st.button("Predict Parts Per Hour"):
        input_data = {
            'Injection_Temperature': injection_temp,
            'Injection_Pressure': injection_pressure,
            'Cycle_Time': cycle_time,
            'Cooling_Time': cooling_time,
            'Material_Viscosity': material_viscosity,
            'Ambient_Temperature': ambient_temp,
            'Machine_Age': machine_age,
            'Operator_Experience': operator_exp,
            'Maintenance_Hours': maintenance_hours,
            'Temperature_Pressure_Ratio': temperature_pressure_ratio,
            'Total_Cycle_Time': total_cycle_time,
            'Efficiency_Score': efficiency_score,
            'Machine_Utilization': machine_utilization,
        }

        input_df = pd.DataFrame([input_data])

        for s in ['Evening', 'Night']:
            input_df[f'Shift_{s}'] = (shift == s)
        for mt in ['Type_B', 'Type_C']:
            input_df[f'Machine_Type_{mt}'] = (machine_type == mt)
        for mg in ['Premium', 'Standard']:
            input_df[f'Material_Grade_{mg}'] = (material_grade == mg)
        for dow in ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']:
            input_df[f'Day_of_Week_{dow}'] = (day_of_week == dow)

        categorical_cols = [
            'Shift_Evening', 'Shift_Night',
            'Machine_Type_Type_B', 'Machine_Type_Type_C',
            'Material_Grade_Premium', 'Material_Grade_Standard',
            'Day_of_Week_Monday', 'Day_of_Week_Saturday',
            'Day_of_Week_Sunday', 'Day_of_Week_Thursday',
            'Day_of_Week_Tuesday', 'Day_of_Week_Wednesday'
        ]

        for col in model_columns:
            if col not in input_df.columns:
                if col in categorical_cols:
                    input_df[col] = False
                else:
                    input_df[col] = 0.0

        input_df = input_df[model_columns]

        trained_cols = scaler.feature_names_in_
        input_df[trained_cols] = scaler.transform(input_df[trained_cols])

        prediction = model.predict(input_df)[0]

        st.markdown(f'<div class="output-text">Predicted Parts Per Hour: {prediction:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-text">Model\'s RMSE: {rmse:.2f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Made with Streamlit · Powered by Your Model</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
