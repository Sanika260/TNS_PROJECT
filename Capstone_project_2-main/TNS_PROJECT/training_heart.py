# Train multiple classifiers on Heart Disease Dataset
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("heart_disease_dataset.csv")
X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale for LR and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train),
    "SVM": SVC(probability=True, random_state=42).fit(X_train_scaled, y_train),
    "DecisionTree": DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
    "RandomForest": RandomForestClassifier(random_state=42).fit(X_train, y_train)
}

# Save models using pickle
for name, model in models.items():
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save scaler
with open("deployed_model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Models trained and saved successfully (using pickle)")
