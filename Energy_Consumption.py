import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Data preprocessing
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Set date as index
train_df.set_index('Date', inplace=True)
test_df.set_index('Date', inplace=True)

# Feature Engineering: Add rolling averages with a smaller window
train_df['Temp_Rolling'] = train_df['Temperature'].rolling(window=3).mean()
train_df['Humidity_Rolling'] = train_df['Humidity'].rolling(window=3).mean()

# Drop rows with NaN values resulting from rolling averages
train_df.dropna(inplace=True)

# Prepare data for modeling
X = train_df[['Temperature', 'Humidity', 'Temp_Rolling', 'Humidity_Rolling']]
y = train_df['Energy_Consumption']

# Use SimpleImputer to handle any remaining NaN values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    results[model_name] = mse
    print(f'{model_name} Mean Squared Error: {mse:.2f}')

# Best model prediction on test data
best_model = min(results, key=results.get)
print(f'Best Model: {best_model}')

# Prepare test data
test_df['Temp_Rolling'] = test_df['Temperature'].rolling(window=3).mean()
test_df['Humidity_Rolling'] = test_df['Humidity'].rolling(window=3).mean()
test_df.dropna(inplace=True)

X_test = test_df[['Temperature', 'Humidity', 'Temp_Rolling', 'Humidity_Rolling']]
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)

# Predict on test data
final_model = models[best_model]
test_predictions = final_model.predict(X_test)

# Save predictions
test_df['Predicted_Energy_Consumption'] = test_predictions
test_df.to_csv('predicted_test.csv', index=True)

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(test_df.index, test_df['Predicted_Energy_Consumption'], label='Predicted')
plt.title('Energy Consumption Prediction')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.savefig('energy_consumption_prediction.png')
plt.show()

# FastAPI implementation

# Define FastAPI app
app = FastAPI()

# Define input data model
class EnergyConsumptionInput(BaseModel):
    Temperature: float
    Humidity: float

# Endpoint for making predictions on new data
@app.post('/predict')
def predict_energy(input_data: EnergyConsumptionInput):
    # Feature engineering for new data
    temp_rolling = input_data.Temperature  # Use the given temperature as a rolling value
    humidity_rolling = input_data.Humidity  # Use the given humidity as a rolling value
    
    # Create input DataFrame
    input_df = pd.DataFrame({
        'Temperature': [input_data.Temperature],
        'Humidity': [input_data.Humidity],
        'Temp_Rolling': [temp_rolling],
        'Humidity_Rolling': [humidity_rolling]
    })
    
    # Impute missing values (if any)
    input_imputed = imputer.transform(input_df)
    
    # Standardize features
    input_scaled = scaler.transform(input_imputed)
    
    # Predict using the best model
    prediction = final_model.predict(input_scaled)
    
    # Return the prediction
    return {'Predicted_Energy_Consumption': prediction[0]}

# Endpoint for predicting on test data
@app.get('/predict/test')
def predict_test_data():
    # Predict on test data
    test_predictions = final_model.predict(X_test)
    
    # Prepare response
    response = test_df[['Temperature', 'Humidity', 'Predicted_Energy_Consumption']].copy()
    response['Predicted_Energy_Consumption'] = test_predictions
    
    return response.to_dict(orient='records')

# Run the app using Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
