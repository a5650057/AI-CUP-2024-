import os
import pandas as pd
import json
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

test_file = '../../../data/final_testing_data/final_test_angle.csv'  

output_dir = 'reappear_test'  
model_output_dir = 'reappear'  
origin_model_output_dir = 'model_output'
training_features_file = os.path.join(origin_model_output_dir, 'training_features.json')
final_models_file = os.path.join(model_output_dir, 'final_models.json')
encoder_filename = os.path.join(origin_model_output_dir, 'onehot_encoder.pkl')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

test_df = pd.read_csv(test_file)

test_df['LocationCode'] = test_df['LocationCode'].astype(str)

test_df['DateTime'] = pd.to_datetime(test_df['DateTime'])
test_df['Hour'] = test_df['DateTime'].dt.hour
test_df['Minute'] = test_df['DateTime'].dt.minute
test_df['Month'] = test_df['DateTime'].dt.month


test_df = test_df.sort_values(['LocationCode', 'DateTime']).reset_index(drop=True)

merged_test_file = os.path.join(output_dir, 'test_withfeature.csv')
test_df.to_csv(merged_test_file, index=False, encoding='utf-8-sig')
print(f"處理後的數據已保存到 {merged_test_file}")

with open(training_features_file, 'r') as f:
    training_features = json.load(f)

print("OneHotEncoder...")
encoder = joblib.load(encoder_filename)

location_encoded = encoder.transform(test_df[['LocationCode']])

location_encoded_cols = encoder.get_feature_names_out(['LocationCode'])

location_encoded_df = pd.DataFrame(location_encoded, columns=location_encoded_cols, index=test_df.index)

train_location_cols = [col for col in training_features if col.startswith('LocationCode_')]
for col in train_location_cols:
    if col not in location_encoded_df.columns:
        location_encoded_df[col] = 0
        print(f"添加缺失的獨熱編碼: {col}")

loc_X_test_base = test_df.drop(columns=['LocationCode', 'DateTime', 'Power(mW)']).copy()
loc_X_test = pd.concat([loc_X_test_base, location_encoded_df], axis=1)

for col in train_location_cols:
    if col not in loc_X_test.columns:
        loc_X_test[col] = 0


try:
    loc_X_test = loc_X_test[training_features]
except KeyError as e:
    missing_cols = list(e.args[0].split(", "))
    for col in missing_cols:
        if col not in loc_X_test.columns:
            loc_X_test[col] = 0
    loc_X_test = loc_X_test[training_features]

if loc_X_test.isnull().values.any():
    loc_X_test = loc_X_test.fillna(0)

with open(final_models_file, 'r') as f:
    final_models = json.load(f)

loaded_models = []
for model_info in final_models:
    model_path = os.path.join(model_output_dir, model_info['xgboost_model'])
    model = XGBRegressor()
    model.load_model(model_path)
    loaded_models.append(model)

print("開始預測...")
predictions = np.zeros(len(loc_X_test))

for model in loaded_models:
    y_pred = model.predict(loc_X_test)
    predictions += y_pred

predictions /= len(loaded_models)

test_df['Predicted_Power(mW)'] = predictions

output_df = test_df[['LocationCode', 'DateTime', 'Predicted_Power(mW)']].copy()

output_predictions_file = os.path.join(output_dir, 'test_predictions.csv')
print(f"保存預測結果到 {output_predictions_file}...")
output_df.to_csv(output_predictions_file, index=False)
print("測試預測完成。")
