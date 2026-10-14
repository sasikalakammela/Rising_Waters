# ==========================================
# Epic 3 - Data Pre-processing
# ==========================================

# Import Libraries
import os
import pandas as pd

# Create output folder
os.makedirs("output", exist_ok=True)

# Read Dataset
df = pd.read_excel("dataset/flood dataset.xlsx")

# ==========================================
# Display Dataset
# ==========================================

print("First 5 Rows:")
print(df.head())

# ==========================================
# Handling Missing Values
# ==========================================

print("\nMissing Values Before Processing:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

print("\nMissing Values After Processing:")
print(df.isnull().sum())

print("\nDataset Shape After Removing Missing Values:")
print(df.shape)
# ==========================================
# Handling Outliers
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Box Plot Before Removing Outliers
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Temp"])
plt.title("Temperature Before Removing Outliers")

plt.savefig("output/outliers_before.png")
plt.show()

# Calculate IQR
Q1 = df["Temp"].quantile(0.25)
Q3 = df["Temp"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remove Outliers
df = df[(df["Temp"] >= lower_limit) & (df["Temp"] <= upper_limit)]

# Box Plot After Removing Outliers
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Temp"])
plt.title("Temperature After Removing Outliers")

plt.savefig("output/outliers_after.png")
plt.show()

print("\nDataset Shape After Removing Outliers:")
print(df.shape)
# ==========================================
# Handling Categorical Values
# ==========================================

print("\nChecking for Categorical Columns...")

categorical_columns = df.select_dtypes(include=["object"]).columns

if len(categorical_columns) == 0:
    print("No categorical columns found.")
else:
    print("Categorical Columns:")
    print(categorical_columns)
    # ==========================================
# Splitting Data into Training and Test Sets
# ==========================================

from sklearn.model_selection import train_test_split

# Separate Features and Target
X = df.drop("flood", axis=1)
y = df["flood"]

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDataset Successfully Split!")

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)
# ==========================================
# Feature Scaling
# ==========================================

from sklearn.preprocessing import StandardScaler

# Create StandardScaler object
scaler = StandardScaler()

# Scale Training Data
X_train_scaled = scaler.fit_transform(X_train)

# Scale Testing Data
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully!")

print("\nScaled Training Data Shape:", X_train_scaled.shape)
print("Scaled Testing Data Shape:", X_test_scaled.shape)

print("\nFirst 5 Rows of Scaled Training Data:")
print(X_train_scaled[:5])