# Grape-yield-prediction-using-GBM
#new
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


data = pd.read_csv('/content/Crop_production.csv')

print(data.head())


X = data.drop('Yield_ton_per_hec', axis=1)
y = data['Yield_ton_per_hec']


categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns


categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])


scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-scores)

print(f'Cross-Validated RMSE: {rmse_scores}')
print(f'Mean RMSE: {rmse_scores.mean()}')
print(f'Standard Deviation of RMSE: {rmse_scores.std()}')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')


joblib.dump(model, 'gbm_grape_yield_model.pkl')

loaded_model = joblib.load('gbm_grape_yield_model.pkl')


new_data = pd.DataFrame({
    'Unnamed: 0':[200],
    'State_Name': ['assam'],
    'Crop_Type': ['whole year'],
    'Crop': ['cotton'],
    'N': [20],
    'P': [20],
    'K': [20],
    'pH': [5],
    'rainfall': [600],
    'temperature': [20],
    'Area_in_hectares': [1000],
    'Production_in_tons': [2000]
})

# Make predictions on new data
new_predictions = loaded_model.predict(new_data)
print("Yield:", new_predictions)
