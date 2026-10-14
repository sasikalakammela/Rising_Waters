# ==========================================
# Epic 4 - Model Building
# Decision Tree Model
# ==========================================

# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Create output folder
os.makedirs("output", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_excel("dataset/flood dataset.xlsx")

print(df.columns)
print(df.head())

# ==========================================
# Features and Target
# ==========================================

X = df.drop("flood", axis=1)
y = df["flood"]

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Decision Tree Model
# ==========================================

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred_dt = decision_tree.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:")
print(accuracy_dt)

# ==========================================
# Classification Report
# ==========================================

print("\nDecision Tree Classification Report:\n")
print(classification_report(y_test, y_pred_dt))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(5,4))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("output/decision_tree_confusion_matrix.png")

plt.show()

# ==========================================
# Random Forest Model
# ==========================================

from sklearn.ensemble import RandomForestClassifier

# Create Model
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
random_forest.fit(X_train, y_train)

# Prediction
y_pred_rf = random_forest.predict(X_test)

# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\n=========================================")
print("Random Forest Accuracy:")
print(accuracy_rf)

# Classification Report
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("output/random_forest_confusion_matrix.png")
plt.show()

# ==========================================
# K-Nearest Neighbors (KNN) Model
# ==========================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Feature Scaling for KNN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Model
knn = KNeighborsClassifier(n_neighbors=5)

# Train Model
knn.fit(X_train_scaled, y_train)

# Prediction
y_pred_knn = knn.predict(X_test_scaled)

# Accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("\n=========================================")
print("KNN Accuracy:")
print(accuracy_knn)

# Classification Report
print("\nKNN Classification Report:\n")
print(classification_report(y_test, y_pred_knn))

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("output/knn_confusion_matrix.png")

plt.show()

# ==========================================
# XGBoost Model
# ==========================================

xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("\n=========================================")
print("XGBoost Accuracy:")
print(accuracy_xgb)

print("\nXGBoost Classification Report:\n")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(5,4))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Purples")

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("output/xgboost_confusion_matrix.png")

plt.show()

# ==========================================
# Compare Models
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "KNN",
        "XGBoost"
    ],
    "Accuracy": [
        accuracy_dt,
        accuracy_rf,
        accuracy_knn,
        accuracy_xgb
    ]
})

# ==========================================
# Save Best Model
# ==========================================

import joblib

joblib.dump(xgb_model, "output/best_model.pkl")
print("XGBoost model saved successfully.")

print("\nBest model saved successfully.")