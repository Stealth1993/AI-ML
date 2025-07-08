# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Using the Palmer Penguins dataset available through Seaborn
df = sns.load_dataset("penguins")

# Step 2: Explore the data
print("First few rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 3: Handle missing values
# Drop rows with missing values for simplicity
df = df.dropna()

# Step 4: Visualize the data
# Create a pairplot to explore relationships between features, colored by species
sns.pairplot(df, hue="species")
plt.show()

# Step 5: Preprocess the data
# Separate features (X) and target (y)
X = df.drop("species", axis=1)  # Features
y = df["species"]               # Target (species to predict)

# One-hot encode categorical columns (island and sex)
categorical_cols = ["island", "sex"]
X = pd.get_dummies(X, columns=categorical_cols)

# Scale numerical columns to standardize them
numerical_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 6: Split the data into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a Random Forest classifier
# Random Forest is chosen for its robustness and ability to handle classification tasks
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions and evaluate the model
# Predict species for the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)

# Create and visualize a confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 9: Analyze feature importances
# Show which features are most important for predicting species
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh")
plt.title("Feature Importances")
plt.show()