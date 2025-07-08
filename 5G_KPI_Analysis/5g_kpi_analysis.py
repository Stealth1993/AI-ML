import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
df = pd.read_csv('5G_KPI_DATA.csv')
df = df.replace('-', np.nan)
df = df[df['NetworkMode'] == '5G']
features = ['Longitude', 'Latitude', 'Speed', 'RSRP', 'RSRQ', 'SNR', 'CQI', 'RSSI', 'State']
target = 'DL_bitrate'
df = df[['Timestamp'] + features + [target]]
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d_%H.%M.%S')
numerical_cols = ['Longitude', 'Latitude', 'Speed', 'RSRP', 'RSRQ', 'SNR', 'CQI', 'RSSI', 'DL_bitrate']
df[numerical_cols] = df[numerical_cols].astype(float)
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])
df = df.dropna()
df = df.sort_values('Timestamp')

# Split the data into training and testing sets (80 percent train, 20 percent test)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size] #  iloc is used for integer-location based indexing
test_df = df.iloc[train_size:]
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizations

# Line Chart: Actual vs. Predicted DL_bitrate over time
plt.figure(figsize=(12, 6))
plt.plot(test_df['Timestamp'], y_test, label='Actual DL_bitrate', color='blue')
plt.plot(test_df['Timestamp'], y_pred, label='Predicted DL_bitrate', color='red')
plt.xlabel('Timestamp')
plt.ylabel('DL_bitrate')
plt.title('Actual vs. Predicted DL_bitrate Over Time')
plt.legend()
plt.show()

# Bar Chart: Feature Importances
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Heatmap: Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df[features + [target]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Histogram: Distribution of DL_bitrate
plt.figure(figsize=(8, 6))
sns.histplot(df['DL_bitrate'], bins=30, kde=True)
plt.title('Distribution of DL_bitrate')
plt.xlabel('DL_bitrate')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot: RSRP vs. DL_bitrate
plt.figure(figsize=(8, 6))
sns.scatterplot(x='RSRP', y='DL_bitrate', data=df)
plt.title('RSRP vs. DL_bitrate')
plt.xlabel('RSRP (dBm)')
plt.ylabel('DL_bitrate (kbps)')
plt.show()