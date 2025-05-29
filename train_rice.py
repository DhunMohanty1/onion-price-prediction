import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
path = r'C:\Users\rkm71\Downloads\price_data (3).csv'
onion = pd.read_csv(path)
onion = onion.dropna()

# Date processing
onion['t'] = pd.to_datetime(onion['t'], errors='coerce')
onion.loc[onion['t'].notnull(), 'day'] = onion.loc[onion['t'].notnull(), 't'].dt.day.astype(int)
onion.loc[onion['t'].notnull(), 'month'] = onion.loc[onion['t'].notnull(), 't'].dt.month.astype(int)
onion.loc[onion['t'].notnull(), 'year'] = onion.loc[onion['t'].notnull(), 't'].dt.year.astype(int)

# One-hot encode 'market_name'
onion_encoded = pd.get_dummies(onion, columns=['market_name'], drop_first=True)

# Ensure day, month, year are integers
onion_encoded['day'] = onion_encoded['day'].astype(int)
onion_encoded['month'] = onion_encoded['month'].astype(int)
onion_encoded['year'] = onion_encoded['year'].astype(int)

# Prepare features and target variable
X = onion_encoded.drop(['p_modal', 't', 'cmdty', 'p_min', 'p_max', 'market_id'], axis=1)
y = onion_encoded['p_modal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Train the model
gbr.fit(X_train, y_train)

# Predict on test set
y_pred = gbr.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Gradient Boosting Regressor Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Save the trained model
joblib.dump(gbr, 'gbr_price_predictor.pkl')
print("Model saved as gbr_price_predictor.pkl")
