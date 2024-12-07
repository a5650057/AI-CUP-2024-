import os
import pandas as pd
import json
import numpy as np
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

data_folder = '../../../data/final_training_data'

all_data = []
for i in range(1, 18):
    file_path = os.path.join(data_folder, f'L{i}_angle_story.csv')
    if not os.path.isfile(file_path):
        print(f"警告: 檔案未找到，跳過: {file_path}")
        continue
    df = pd.read_csv(file_path)
    all_data.append(df)

if not all_data:
    raise ValueError("沒有讀取任何訓練資料檔案。請檢查檔案路徑和檔名。")

merged_df = pd.concat(all_data, ignore_index=True)
merged_df = merged_df[['LocationCode', 'DateTime','H_UVI_TEMP_interaction','GHI','Azimuth','Elevation','Solarinsolation','angle','story','DHI','DNI','Power(mW)']]

merged_df['LocationCode'] = merged_df['LocationCode'].astype(str)

print("提取時間特徵...")
merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
merged_df['Hour'] = merged_df['DateTime'].dt.hour
merged_df['Minute'] = merged_df['DateTime'].dt.minute  
merged_df['Month'] = merged_df['DateTime'].dt.month

merged_df = merged_df.sort_values(['LocationCode', 'DateTime']).reset_index(drop=True)

original_encoder_path = 'model_output/onehot_encoder.pkl'
encoder = joblib.load(original_encoder_path)
print("載入原先的 OneHotEncoder：", original_encoder_path)

print("進行獨熱編碼 'LocationCode'...")
location_encoded = encoder.transform(merged_df[['LocationCode']])
location_encoded_cols = encoder.get_feature_names_out(['LocationCode'])
location_encoded_df = pd.DataFrame(location_encoded, columns=location_encoded_cols, index=merged_df.index)
merged_df = pd.concat([merged_df, location_encoded_df], axis=1)
merged_df.drop(columns=['LocationCode', 'DateTime'], inplace=True)

print("準備特徵和標籤")
X = merged_df.drop(columns=['Power(mW)'])
y = merged_df['Power(mW)']

print("檢查是否有任何 NaN 值...")
print(X.isnull().sum())

print("刪除含有 NaN 的行..")
X.dropna(inplace=True)
y = y[X.index]

best_params_path = os.path.join('model_output', 'best_params.json')
with open(best_params_path, 'r') as f:
    best_params = json.load(f)
print("載入最佳參數：", best_params)

fold_indices_path = os.path.join('model_output', 'fold_indices.npy')
fold_indices = np.load(fold_indices_path, allow_pickle=True)
print("載入原先的折疊索引：", fold_indices_path)

reappear_dir = 'reappear'
if not os.path.exists(reappear_dir):
    os.makedirs(reappear_dir)

def evaluate_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2': r2}

final_models = []
oof_preds = np.zeros(len(X))

print("開始使用最佳參數重現訓練過程...")
for fold, indices in enumerate(fold_indices, 1):
    train_idx = indices['train_idx']
    valid_idx = indices['valid_idx']

    X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBRegressor(**best_params)

    model.fit(
        X_train_cv, y_train_cv,
        eval_set=[(X_valid_cv, y_valid_cv)],
        verbose=False
    )

    xgb_filename = f'xgboost_reappear_fold_{fold}.json'
    model.save_model(os.path.join(reappear_dir, xgb_filename))
    print(f"XGBoost 模型已保存至 '{xgb_filename}'")

    final_models.append({'xgboost_model': xgb_filename})

    y_valid_pred = model.predict(X_valid_cv)
    oof_preds[valid_idx] = y_valid_pred

np.save(os.path.join(reappear_dir, 'oof_preds.npy'), oof_preds)
print("OOF 預測已保存。")

metrics_list = []
for fold, model_info in enumerate(final_models, 1):
    model = XGBRegressor()
    model.load_model(os.path.join(reappear_dir, model_info['xgboost_model']))

    valid_idx = fold_indices[fold - 1]['valid_idx']
    X_valid_cv = X.iloc[valid_idx]
    y_valid_cv = y.iloc[valid_idx]

    y_valid_pred = model.predict(X_valid_cv)
    metrics = evaluate_metrics(y_valid_cv, y_valid_pred)
    metrics_list.append(metrics)

avg_metrics = {
    'RMSE': np.mean([m['rmse'] for m in metrics_list]),
    'MSE': np.mean([m['mse'] for m in metrics_list]),
    'MAE': np.mean([m['mae'] for m in metrics_list]),
    'R2': np.mean([m['r2'] for m in metrics_list]),
}

print(f"重現模型的平均 RMSE: {avg_metrics['RMSE']}")
print(f"重現模型的平均 MSE: {avg_metrics['MSE']}")
print(f"重現模型的平均 MAE: {avg_metrics['MAE']}")
print(f"重現模型的平均 R²: {avg_metrics['R2']}")

final_results = {
    'best_params': best_params,
    'average_metrics': avg_metrics,
    'fold_metrics': metrics_list
}

with open(os.path.join(reappear_dir, 'model_results_cv.json'), 'w') as f:
    json.dump(final_results, f, indent=4)

print("重現訓練的結果已保存至 'model_results_cv.json'。")
