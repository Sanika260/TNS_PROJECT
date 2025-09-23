# EDA for Heart Disease Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
df = pd.read_csv("heart_disease_dataset.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nInfo:")
print(df.info())
print("\nSummary:")
print(df.describe())

# Target distribution
sns.countplot(x="heart_disease", data=df, palette="Set2")
plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
plt.show()

# Age distribution
sns.histplot(df["age"], bins=20, kde=True, color="blue")
plt.title("Age Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# T-test (Age: Disease vs No)
age_disease = df[df["heart_disease"]==1]["age"]
age_no_disease = df[df["heart_disease"]==0]["age"]
t_stat, p_val = stats.ttest_ind(age_disease, age_no_disease)
print(f"T-test Age -> t={t_stat:.3f}, p={p_val:.3f}")

# Chi-square (Sex vs Heart Disease)
contingency = pd.crosstab(df["sex"], df["heart_disease"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square Sex vs Heart Disease -> chi2={chi2:.3f}, p={p:.3f}")
