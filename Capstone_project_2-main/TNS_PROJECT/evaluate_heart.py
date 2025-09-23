# Evaluate trained models (Pickle version)
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

# Load dataset
df = pd.read_csv("heart_disease_dataset.csv")
X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

# Load scaler + models
with open("deployed_model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

models = {}
for name in ["LogisticRegression", "SVM", "DecisionTree", "RandomForest"]:
    with open(f"{name}.pkl", "rb") as f:
        models[name] = pickle.load(f)

# Scale data for LR & SVM
X_scaled = scaler.transform(X)

# Evaluate models
results = []
for name, model in models.items():
    if name in ["LogisticRegression", "SVM"]:
        X_use = X_scaled
    else:
        X_use = X

    y_pred = model.predict(X_use)
    y_prob = model.predict_proba(X_use)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_prob) if y_prob is not None else np.nan

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc)
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred, zero_division=0))

    results.append([name, acc, prec, rec, f1, roc])

# Comparison table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
print("\n=== Comparison Table ===")
print(results_df)
