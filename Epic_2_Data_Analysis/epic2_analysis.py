# ==========================================
# Epic 2 - Data Visualization and Analysis
# Rising Waters - Flood Prediction Project
# ==========================================

# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder automatically
os.makedirs("output", exist_ok=True)

# ==========================================
# Read Dataset
# ==========================================

df = pd.read_excel("dataset/flood dataset.xlsx")

# ==========================================
# Display Dataset Information
# ==========================================

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# ==========================================
# Univariate Analysis - Histogram
# ==========================================

plt.figure(figsize=(8,5))
plt.hist(df["Temp"], bins=5)

plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")

plt.savefig("output/temperature_distribution.png")
plt.show()

# ==========================================
# Univariate Analysis - Count Plot
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(x="flood", data=df)

plt.title("Flood Distribution")
plt.xlabel("Flood")
plt.ylabel("Count")

plt.savefig("output/flood_distribution.png")
plt.show()

# ==========================================
# Univariate Analysis - Box Plot
# ==========================================

plt.figure(figsize=(6,4))
sns.boxplot(y=df["Temp"])

plt.title("Temperature Box Plot")

plt.savefig("output/temperature_boxplot.png")
plt.show()

# ==========================================
# Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Correlation Matrix
# ==========================================

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.savefig("output/correlation_matrix.png")
plt.show()

# ==========================================
# Separate Features and Target
# ==========================================

X = df.drop("flood", axis=1)
y = df["flood"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# ==========================================
# Train-Test Split
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# Model Training
# ==========================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed.")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)

# ==========================================
# Accuracy
# ==========================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# ==========================================
# Confusion Matrix
# ==========================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("output/confusion_matrix.png")
plt.show()

# ==========================================
# Classification Report
# ==========================================

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
