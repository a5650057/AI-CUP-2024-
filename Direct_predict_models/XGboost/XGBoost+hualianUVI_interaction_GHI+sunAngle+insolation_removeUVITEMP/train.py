import os
import pandas as pd
import json
import numpy as np
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.preprocessing import OneHotEncoder
import joblib

data_folder = '../../../data/final_training_data_GHI_SunAngle_Solarinsolation'

all_data = []

for i in range(1, 18):
    file_path = os.path.join(data_folder, f'L{i}.csv')
    if not os.path.isfile(file_path):
        continue
    df = pd.read_csv(file_path)
    all_data.append(df)

if not all_data:
    raise ValueError("沒有讀取任何訓練資料檔案。請檢查檔案路徑和檔名")

merged_df = pd.concat(all_data, ignore_index=True)

merged_df = merged_df[['LocationCode', 'DateTime','H_UVI_TEMP_interaction','GHI','Azimuth','Elevation','Solarinsolation','Power(mW)']]

merged_df['LocationCode'] = merged_df['LocationCode'].astype(str)

print("提取時間特徵")
merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
merged_df['Hour'] = merged_df['DateTime'].dt.hour
merged_df['Minute'] = merged_df['DateTime'].dt.minute  
merged_df['Month'] = merged_df['DateTime'].dt.month

merged_df = merged_df.sort_values(['LocationCode', 'DateTime']).reset_index(drop=True)

print("進行獨熱編碼 'LocationCode'...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
location_encoded = encoder.fit_transform(merged_df[['LocationCode']])
location_encoded_cols = encoder.get_feature_names_out(['LocationCode'])
location_encoded_df = pd.DataFrame(location_encoded, columns=location_encoded_cols, index=merged_df.index)
merged_df = pd.concat([merged_df, location_encoded_df], axis=1)

merged_df.drop(columns=['LocationCode', 'DateTime'], inplace=True)

print("計算特徵和目標變數之間的相關度...")
correlation_matrix = merged_df.corr().round(2)

output_dir = 'model_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

correlation_matrix.to_csv(os.path.join(output_dir, 'feature_correlations.csv'))
print("特徵和目標變數的相關度矩陣已儲存至 'feature_correlations.csv'")


print("準備特徵和標籤...")
X = merged_df.drop(columns=['Power(mW)'])
y = merged_df['Power(mW)']

print("檢查是否有任何 NaN 值...")
print(X.isnull().sum())

print("刪除含有 NaN 的行..")
X.dropna(inplace=True)
y = y[X.index] 


output_dir = 'model_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


training_features = X.columns.tolist()
with open(os.path.join(output_dir, 'training_features.json'), 'w') as f:
    json.dump(training_features, f, indent=4)
print("訓練特徵已儲存至 'training_features.json'")

encoder_filename = 'onehot_encoder.pkl'
joblib.dump(encoder, os.path.join(output_dir, encoder_filename))
print(f"OneHotEncoder 已保存至 '{encoder_filename}'")

def evaluate_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2': r2}

def objective(trial: Trial):
    params = {
        "n_estimators": 900,  
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "verbosity": 0,
        'early_stopping_rounds': 50,
        'tree_method': "hist",
        'device': 'cuda', 
    }

    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rmse_scores = []
    
    for train_idx, valid_idx in kf.split(X):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)

        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_valid_cv, y_valid_cv)],
            verbose=False
        )

        y_pred = model.predict(X_valid_cv)

        rmse = mean_squared_error(y_valid_cv, y_pred, squared=False)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50, timeout=21600)  

best_params = study.best_params
best_params["n_estimators"] = 900  
best_params["random_state"] = 42
best_params["verbosity"] = 0
best_params["tree_method"] = "hist"
best_params['device'] = 'cuda' 
best_params["early_stopping_rounds"] = 50

print("最佳超参数：", best_params)

with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=4)

final_models = []
n_splits = 10  
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_indices = []
oof_preds = np.zeros(len(X))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
    X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBRegressor(**best_params)

    model.fit(
        X_train_cv, y_train_cv,
        eval_set=[(X_valid_cv, y_valid_cv)],
        verbose=False
    )

    xgb_filename = f'xgboost_final_fold_{fold}.json'
    model.save_model(os.path.join(output_dir, xgb_filename))
    print(f"XGBoost 模型已保存至 '{xgb_filename}'")

    final_models.append({
        'xgboost_model': xgb_filename
    })

    fold_indices.append({'train_idx': train_idx.tolist(), 'valid_idx': valid_idx.tolist()})

    y_valid_pred = model.predict(X_valid_cv)
    oof_preds[valid_idx] = y_valid_pred

with open(os.path.join(output_dir, 'final_models.json'), 'w') as f:
    json.dump(final_models, f, indent=4)
print("所有折疊的模型已記錄至 'final_models.json'")

np.save(os.path.join(output_dir, 'fold_indices.npy'), fold_indices)

np.save(os.path.join(output_dir, 'oof_preds.npy'), oof_preds)
print("折疊索引和 OOF 預測已保存。")

metrics_list = []
for fold, model_info in enumerate(final_models, 1):
    model = XGBRegressor()
    model.load_model(os.path.join(output_dir, model_info['xgboost_model']))

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

print(f"最終模型的平均 RMSE: {avg_metrics['RMSE']}")
print(f"最終模型的平均 MSE: {avg_metrics['MSE']}")
print(f"最終模型的平均 MAE: {avg_metrics['MAE']}")
print(f"最終模型的平均 R²: {avg_metrics['R2']}")

final_results = {
    'best_params': best_params,
    'average_metrics': avg_metrics,
    'fold_metrics': metrics_list
}

with open(os.path.join(output_dir, 'model_results_cv.json'), 'w') as f:
    json.dump(final_results, f, indent=4)

print("模型的最佳參數和最終評估指標已保存至 'model_results_cv.json'")
