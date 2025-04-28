import joblib
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

model = joblib.load('eth_fee_model_v2.pkl')

# Usuniecie feature names
booster = model.get_booster()
booster.feature_names = None

# Definicja inputu
initial_type = [('float_input', FloatTensorType([None, 8]))]

# Konwersja model
onnx_model = convert_xgboost(booster, initial_types=initial_type)

onnxmltools.utils.save_model(onnx_model, 'eth_fee_model_v2.onnx')

print('✅ Model przekonwertowany poprawnie do ONNX')
