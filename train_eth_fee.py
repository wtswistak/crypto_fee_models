import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

df = pd.read_csv('fee_snapshot.csv').sort_values('created_at')

# skala do Gwei
for col in ['base_fee_per_gas', 'priority_fee_10', 'priority_fee_50', 'priority_fee_90']:
    df[col] /= 1e9

# rolling cechy
df['base_fee_mean_5'] = df['base_fee_per_gas'].rolling(5).mean()
df['base_fee_std_5']  = df['base_fee_per_gas'].rolling(5).std()
df['priority_gap']    = df['priority_fee_90'] - df['priority_fee_10']
df = df.dropna()

# TARGET – 6 kolumn (+10 … +60 min)
for step in range(1, 7):
    df[f'fee_t{step}'] = df['base_fee_per_gas'].shift(-step)

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
    'priority_gap'
]
targets  = [f'fee_t{i}' for i in range(1,7)]

X, y = df[features], df[targets]
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, shuffle=False)

base = XGBRegressor(
    n_estimators=1200, max_depth=8, learning_rate=0.03,
    subsample=0.9, colsample_bytree=0.9, random_state=42
)
model = MultiOutputRegressor(base)
model.fit(Xtr, ytr)

print('MAE / horizon:',
      mean_absolute_error(yte, model.predict(Xte), multioutput='raw_values'))

joblib.dump(model, 'fee_multi6.pkl')
