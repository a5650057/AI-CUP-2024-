import os
import pandas as pd
import json
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_folder = '../../../data/final_training_data'
test_file = '../../../data/final_testing_data/final_test_angle_story.csv'

model_output_dir = 'model_output'  
reappear_output_dir = 'reappear'   

if not os.path.exists(reappear_output_dir):
    os.makedirs(reappear_output_dir)

all_data = []
for i in range(1, 18):
    file_path = os.path.join(data_folder, f'L{i}.csv')
    if not os.path.isfile(file_path):
        continue
    df = pd.read_csv(file_path)
    all_data.append(df)

if not all_data:
    raise ValueError("沒有訓練資料，請檢查路徑")

merged_df = pd.concat(all_data, ignore_index=True)
merged_df = merged_df[['LocationCode', 'DateTime','H_UVI_TEMP_interaction','GHI','Azimuth','Elevation','Solarinsolation','angle','Power(mW)']]

merged_df['LocationCode'] = merged_df['LocationCode'].astype(str)

print("提取時間特徵")
merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
merged_df['Hour'] = merged_df['DateTime'].dt.hour
merged_df['Minute'] = merged_df['DateTime'].dt.minute
merged_df['Month'] = merged_df['DateTime'].dt.month
merged_df = merged_df.sort_values(['LocationCode', 'DateTime']).reset_index(drop=True)

print("應用原始獨熱編碼器")
encoder_filename = 'onehot_encoder.pkl'
encoder = joblib.load(os.path.join(model_output_dir, encoder_filename))

location_encoded = encoder.transform(merged_df[['LocationCode']])
location_encoded_cols = encoder.get_feature_names_out(['LocationCode'])
location_encoded_df = pd.DataFrame(location_encoded, columns=location_encoded_cols, index=merged_df.index)
merged_df = pd.concat([merged_df, location_encoded_df], axis=1)
merged_df.drop(columns=['LocationCode', 'DateTime'], inplace=True)

print("準備特徵和標籤")
X = merged_df.drop(columns=['Power(mW)'])
y = merged_df['Power(mW)']

print("檢查是否有任何 NaN 值")
if X.isnull().values.any():
    print("發現 NaN 值，填充為 0")
    X = X.fillna(0)

with open(os.path.join(model_output_dir, 'training_features.json'), 'r') as f:
    training_features = json.load(f)

for col in training_features:
    if col not in X.columns:
        X[col] = 0
X = X[training_features]

with open(os.path.join(model_output_dir, 'best_params.json'), 'r') as f:
    best_params = json.load(f)

fold_indices = np.load(os.path.join(model_output_dir, 'fold_indices.npy'), allow_pickle=True)

print("加載原始模型並進行評估")
metrics_list = []
final_models = []
oof_preds = np.zeros(len(X))

for fold, indices in enumerate(fold_indices, 1):
    train_idx = indices['train_idx']
    valid_idx = indices['valid_idx']

    X_valid_cv = X.iloc[valid_idx]
    y_valid_cv = y.iloc[valid_idx]

    xgb_filename = f'xgboost_final_fold_{fold}.json'
    model_path = os.path.join(model_output_dir, xgb_filename)
    model = XGBRegressor()
    model.load_model(model_path)

    reappear_model_path = os.path.join(reappear_output_dir, xgb_filename)
    model.save_model(reappear_model_path)
    print(f"XGBoost 模型已保存到 '{reappear_model_path}'")

    final_models.append({
        "xgboost_model": xgb_filename
    })

    y_valid_pred = model.predict(X_valid_cv)
    oof_preds[valid_idx] = y_valid_pred

    
    metrics = {
        'rmse': mean_squared_error(y_valid_cv, y_valid_pred, squared=False),
        'mse': mean_squared_error(y_valid_cv, y_valid_pred),
        'mae': mean_absolute_error(y_valid_cv, y_valid_pred),
        'r2': r2_score(y_valid_cv, y_valid_pred)
    }
    metrics_list.append(metrics)

avg_metrics = {
    'RMSE': np.mean([m['rmse'] for m in metrics_list]),
    'MSE': np.mean([m['mse'] for m in metrics_list]),
    'MAE': np.mean([m['mae'] for m in metrics_list]),
    'R2': np.mean([m['r2'] for m in metrics_list]),
}
print(f"平均 RMSE: {avg_metrics['RMSE']}, 平均 MAE: {avg_metrics['MAE']}")

with open(os.path.join(reappear_output_dir, 'final_models.json'), 'w') as f:
    json.dump(final_models, f, indent=4)
print("模型列表已保存到 'final_models.json'")

final_results = {
    'best_params': best_params,
    'average_metrics': avg_metrics,
    'fold_metrics': metrics_list
}
with open(os.path.join(reappear_output_dir, 'model_results_cv.json'), 'w') as f:
    json.dump(final_results, f, indent=4)
print("結果已保存到 'model_results_cv.json'")
