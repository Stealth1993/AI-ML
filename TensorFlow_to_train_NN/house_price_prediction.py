# This program uses TensorFlow to train a neural network for predicting house prices
# based on the California Housing dataset. It includes data preprocessing, model training,
# evaluation, comparison with linear regression, and visualization of training progress.

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: Disable oneDNN for reproducibility
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target  # Target is the median house value in $100,000

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model with dynamic input shape
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),  # Dynamically set input shape
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model with built-in RMSE metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model on the test set
test_loss, test_rmse = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Neural Network Test RMSE: {test_rmse:.2f} (in $100,000)")

# Train a linear regression model for comparison
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
print(f"Linear Regression Test RMSE: {rmse_lin:.2f} (in $100,000)")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model with .keras extension
model.save('house_price_model.keras')

# Make a sample prediction
sample = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(sample)
print(f"Predicted house price: {prediction[0][0]:.2f} (in $100,000)")
print(f"Actual house price: {y_test[0]:.2f} (in $100,000)")