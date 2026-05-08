import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import pickle
#import dataset
df=pd.read_csv("Walmart_Sales.csv")

print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())

# convert date to datetime and extract year, month, week
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

# EDA
df.drop('Date', axis=1, inplace=True)

# Distribution of Weekly Sales
plt.figure(figsize=(10,6))
sns.histplot(df['Weekly_Sales'], kde=True)
plt.title('Distribution of Weekly Sales')
plt.show()

# Monthly Sales Comparison
plt.figure(figsize=(10,6))
sns.barplot(x='Month', y='Weekly_Sales', data=df)
plt.title('Monthly Sales Comparison')
plt.show()

# Holiday vs Non-Holiday Sales
plt.figure(figsize=(8,6))
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title('Holiday vs Non-Holiday Sales')
plt.show()

# Temperature vs Sales
plt.figure(figsize=(10,6))
sns.scatterplot(x='Temperature', y='Weekly_Sales', data=df)
plt.title('Temperature vs Weekly Sales')
plt.show()

# Fuel Price vs Sales
plt.figure(figsize=(10,6))
sns.scatterplot(x='Fuel_Price', y='Weekly_Sales', data=df)
plt.title('Fuel Price vs Weekly Sales')
plt.show()

# dependence of features
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Prepare data for modeling
X = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

#random forest regression
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rf_r2 = r2_score(y_test, y_pred)

print('MAE:', mae)
print('RMSE:', rmse)
print('R2 Score:', rf_r2)


# Feature Importance

importance = rf.feature_importances_
features = X.columns

feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance')
plt.show()




ann = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])


# Compile ANN
ann.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train ANN
history = ann.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ANN PREDICTIONS

ann_pred = ann.predict(X_test_scaled)

# ANN EVALUATION


ann_mae = mean_absolute_error(y_test, ann_pred)

ann_rmse = np.sqrt(
    mean_squared_error(y_test, ann_pred)
)

ann_r2 = r2_score(y_test, ann_pred)

print("\nANN Results")

print("MAE:", ann_mae)
print("RMSE:", ann_rmse)
print("R2 Score:", ann_r2)


# MODEL COMPARISON GRAPH


models = [
    'Random Forest',
    'ANN'
]

r2_scores = [
    rf_r2,
    ann_r2
]

plt.figure(figsize=(8,5))

sns.barplot(
    x=models,
    y=r2_scores
)

plt.title('Model Performance Comparison')
plt.ylabel('R2 Score')

plt.show()

# ANN LOSS CURVE


plt.figure(figsize=(10,6))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('ANN Loss Curve')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
pickle.dump(rf, open("model.pkl", "wb"))

print("\nModel saved successfully as model.pkl")