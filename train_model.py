import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

path = r'C:\Users\rkm71\Downloads\price_data (2).csv'
onion = pd.read_csv(path)
onion = onion.dropna()

# Your new block added here exactly as you wrote it:
onion['t'] = pd.to_datetime(onion['t'], errors='coerce')
onion.loc[onion['t'].notnull(), 'day'] = onion.loc[onion['t'].notnull(), 't'].dt.day.astype(int)
onion.loc[onion['t'].notnull(), 'month'] = onion.loc[onion['t'].notnull(), 't'].dt.month.astype(int)
onion.loc[onion['t'].notnull(), 'year'] = onion.loc[onion['t'].notnull(), 't'].dt.year.astype(int)
onion_encoded = pd.get_dummies(onion, columns=['market_name'], drop_first=True)
onion['day'] = onion['day'].astype(int)
onion['month'] = onion['month'].astype(int)
onion['year'] = onion['year'].astype(int)
onion

from sklearn.model_selection import train_test_split
# Features (you can add/remove features as needed)
X = onion_encoded.drop(['p_modal', 't', 'cmdty', 'p_min', 'p_max','market_id'], axis=1)

# Target
y = onion_encoded['p_modal']
t = onion_encoded['t']  # datetime column

X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, t, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model Evaluation Results:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

import joblib

# After training the model
joblib.dump(model, 'price_predictor.pkl')
print("Model saved as price_predictor.pkl")
