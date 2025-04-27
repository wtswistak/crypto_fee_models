import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('fee_snapshot.csv')

# Skalowanie danych (Wei -> Gwei)
SCALE = 1e9
for col in ['base_fee_per_gas', 'priority_fee_10', 'priority_fee_50', 'priority_fee_90']:
    df[col] = df[col] / SCALE

df = df.sort_values(by='created_at')

# Feature Engineering
df['base_fee_mean_5'] = df['base_fee_per_gas'].rolling(window=5).mean()
df['base_fee_std_5'] = df['base_fee_per_gas'].rolling(window=5).std()
df['priority_fee_gap'] = df['priority_fee_90'] - df['priority_fee_10']

# Zmienna docelowa (future baseFee)
df['future_base_fee'] = df['base_fee_per_gas'].shift(-1)

df = df.dropna()

# Definicja cech (features) i cel (target)
features = [
    'base_fee_per_gas',
    'gas_used_ratio',
    'priority_fee_10',
    'priority_fee_50',
    'priority_fee_90',
    'base_fee_mean_5',
    'base_fee_std_5',
    'priority_fee_gap'
]

X = df[features]
y = df['future_base_fee']

# Podział dannych na zbior treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Stworzenie i trenowanie model
model = XGBRegressor(
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=1
)

print('Trening modelu')
model.fit(X_train, y_train)

print('Ocena modelu')
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f} Gwei')

dest_path = 'eth_fee_model_v2.pkl'
joblib.dump(model, dest_path)
print(f'Model zapisany do pliku: {dest_path}')
