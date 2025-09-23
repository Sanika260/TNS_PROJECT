import streamlit as st
import pandas as pd
import pickle

# load models
model = pickle.load(open("deployed_model/RandomForest.pkl", "rb"))
scaler = pickle.load(open("deployed_model/scaler.pkl", "rb"))

st.title("Heart Disease Prediction")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0,1])
cp = st.number_input("Chest Pain Type", 0, 3, 0)
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
restecg = st.selectbox("Resting ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.number_input("Major Vessels", 0, 3, 0)
thal = st.selectbox("Thalassemia", [1,2,3])

if st.button("Predict"):
    X = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
                     columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.success("Heart Disease" if pred==1 else "No Heart Disease")
