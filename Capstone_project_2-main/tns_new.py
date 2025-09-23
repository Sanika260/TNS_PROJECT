import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("TNS_PROJECT/heart_disease_dataset.csv")

X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=42
)

# scaling for LR and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train),
    "SVM": SVC(probability=True, random_state=42).fit(X_train_scaled, y_train),
    "Decision Tree": DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
    "Random Forest": RandomForestClassifier(random_state=42).fit(X_train, y_train)
}

def evaluate_model(name, model):
    if name in ["Logistic Regression", "SVM"]:
        X_test_use = X_test_scaled
    else:
        X_test_use = X_test

    y_pred = model.predict(X_test_use)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_use)[:, 1]
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)

    report = classification_report(y_test, y_pred, zero_division=0)

    result_text = f"""
=== {name} ===
Accuracy     : {acc:.4f}
Precision    : {prec:.4f}
Recall       : {rec:.4f}
F1 Score     : {f1:.4f}
ROC-AUC      : {roc:.4f}
Specificity  : {spec:.4f}

Confusion Matrix:
{confusion_matrix(y_test, y_pred)}

Classification Report:
{report}
"""
    return result_text, [acc, prec, rec, f1, roc, spec]

# Evaluate all models for summary
summary_results = {}
for name, model in models.items():
    _, metrics = evaluate_model(name, model)
    summary_results[name] = metrics

summary_df = pd.DataFrame(summary_results, index=["Accuracy","Precision","Recall","F1","ROC-AUC","Specificity"]).T

# ----------------------------
# GUI Layout
# ----------------------------
root = tk.Tk()
root.title("Heart Disease Detection - Classification App")
root.geometry("1100x650")

# Create main frames
left_frame = tk.Frame(root, padx=15, pady=15, bg="#f0f0f0")
left_frame.pack(side="left", fill="y")

right_frame = tk.Frame(root, padx=10, pady=10)
right_frame.pack(side="right", fill="both", expand=True)

# ----------------------------
# Left: Controls + Input
# ----------------------------
label = tk.Label(left_frame, text="Select Model:", font=("Arial", 12), bg="#f0f0f0")
label.pack(pady=5)

selected_model = tk.StringVar()
model_dropdown = ttk.Combobox(left_frame, textvariable=selected_model, state="readonly",
                              values=list(models.keys()))
model_dropdown.pack(pady=5, fill="x")
model_dropdown.current(0)

btn_model = tk.Button(left_frame, text="Show Model Details", command=lambda: show_model_details(),
                      width=25, bg="lightblue")
btn_model.pack(pady=5)

btn_summary = tk.Button(left_frame, text="Summary Report", command=lambda: show_summary(),
                        width=25, bg="lightgreen")
btn_summary.pack(pady=5)

# Patient input section
param_label = tk.Label(left_frame, text="Enter Patient Data:", font=("Arial", 12, "bold"), bg="#f0f0f0")
param_label.pack(pady=10)

param_frame = tk.Frame(left_frame, bg="#f0f0f0")
param_frame.pack()

feature_names = list(X.columns)
entry_boxes = {}

for i, feat in enumerate(feature_names):
    lbl = tk.Label(param_frame, text=feat, width=20, anchor="w", bg="#f0f0f0")
    lbl.grid(row=i, column=0, pady=2, sticky="w")
    ent = tk.Entry(param_frame, width=10)
    ent.grid(row=i, column=1, pady=2)
    entry_boxes[feat] = ent

btn_predict = tk.Button(left_frame, text="Predict Patient", command=lambda: predict_patient(),
                        width=25, bg="orange")
btn_predict.pack(pady=10)

# ----------------------------
# Right: Output area
# ----------------------------
output_area = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, width=90, height=35, font=("Courier", 10))
output_area.pack(fill="both", expand=True)

# ----------------------------
# Functions
# ----------------------------
def show_model_details():
    model_name = selected_model.get()
    model = models[model_name]
    text, _ = evaluate_model(model_name, model)
    output_area.delete(1.0, tk.END)
    output_area.insert(tk.END, text)

def show_summary():
    output_area.delete(1.0, tk.END)
    output_area.insert(tk.END, "=== Summary Report: Model Comparison ===\n\n")
    output_area.insert(tk.END, str(summary_df.round(4)))

def predict_patient():
    try:
        model_name = selected_model.get()
        model = models[model_name]

        values = []
        for feat in feature_names:
            val = float(entry_boxes[feat].get())
            values.append(val)

        arr = np.array(values).reshape(1, -1)

        if model_name in ["Logistic Regression", "SVM"]:
            arr = scaler.transform(arr)

        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0,1] if hasattr(model, "predict_proba") else None

        result = f"Model: {model_name}\nPrediction: {'Heart Disease' if pred==1 else 'No Heart Disease'}"
        if prob is not None:
            result += f"\nProbability of Disease: {prob:.4f}"

        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# ----------------------------
# Run GUI
# ----------------------------
root.mainloop()



os.makedirs("deployed_model", exist_ok=True)

# Save Random Forest model (from your models dict)
rf_model = models.get("Random Forest")
if rf_model is not None:
    joblib.dump(rf_model, os.path.join("deployed_model", "RandomForest.pkl"))
    print("Saved RandomForest.pkl")

# Save scaler (you created scaler earlier)
try:
    joblib.dump(scaler, os.path.join("deployed_model", "scaler.pkl"))
    print("Saved scaler.pkl")
except Exception as e:
    print("Could not save scaler:", e)

# Save feature names (important: same order as training)
feature_names = list(X.columns)
with open(os.path.join("deployed_model", "feature_names.json"), "w") as f:
    json.dump(feature_names, f)
print("Saved feature_names.json")