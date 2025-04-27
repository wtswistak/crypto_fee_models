import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('fee_snapshot.csv')

SCALE = 1e9  # Zamiana z Wei na Gwei

df['base_fee_per_gas'] = df['base_fee_per_gas'] / SCALE
df['priority_fee_10'] = df['priority_fee_10'] / SCALE
df['priority_fee_50'] = df['priority_fee_50'] / SCALE
df['priority_fee_90'] = df['priority_fee_90'] / SCALE

df = df.sort_values(by='created_at')

# Zmienna docelowa (przyszłe baseFeePerGas)
df['future_base_fee'] = df['base_fee_per_gas'].shift(-1)

# Usuwamy ostatni rekord (nie ma dla niego "przyszłości")
df = df.dropna()

# Cechy (features) i cel (target)
features = [
    'base_fee_per_gas',
    'gas_used_ratio',
    'priority_fee_10',
    'priority_fee_50',
    'priority_fee_90'
]

X = df[features]
y = df['future_base_fee']

# Podział dannych na zbior treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # NIE mieszamy czasowych danych!
)

# stworzenie i trenowanie model
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print('Trening modelu')
model.fit(X_train, y_train)

print('Ocena modelu')
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

dest_path = 'eth_fee_model.pkl'
joblib.dump(model, dest_path)
print(f'Model zapisany do pliku: {dest_path}')
