import pandas as pd

test_without_ghi_path = "reappear_test/test_withfeature.csv"
test_predictions_path = "reappear_test/test_predictions.csv"
output_path = "../../../data/XGBoost_pseudo_labels_traingdata/test_pseudo.csv"

test_without_ghi = pd.read_csv(test_without_ghi_path)
test_predictions = pd.read_csv(test_predictions_path)

test_without_ghi['DateTime'] = pd.to_datetime(test_without_ghi['DateTime'])
test_predictions['DateTime'] = pd.to_datetime(test_predictions['DateTime'])

test_predictions['Predicted_Power(mW)'] = test_predictions['Predicted_Power(mW)'].apply(lambda x: max(x, 0))

merged_df = pd.merge(test_without_ghi, test_predictions, on=['LocationCode', 'DateTime'], how='left')

merged_df['Power(mW)'] = merged_df['Power(mW)'].fillna(merged_df['Predicted_Power(mW)'])

merged_df = merged_df.drop(columns=['Predicted_Power(mW)'])

merged_df.to_csv(output_path, index=False)
print(f"合併完成，結果已儲存至 {output_path}")
