import pandas as pd
import os

sun_data_path = 'sun_Azimuth_elevation_angle.csv'
sun_data = pd.read_csv(sun_data_path)
sun_data['DateTime'] = pd.to_datetime(sun_data['DateTime'])

training_source_folder = 'data/remove_duplicate/'
testing_source_folder = 'data/final_testing_data/'

training_target_folder = 'data/final_training_data/'
testing_target_folder = 'data/final_testing_data/'

os.makedirs(training_target_folder, exist_ok=True)
os.makedirs(testing_target_folder, exist_ok=True)

for i in range(1, 18):
    source_file = os.path.join(training_source_folder, f'L{i}_Train_with_hualianData.csv')
    if not os.path.exists(source_file):
        print(f"訓練檔案不存在：{source_file}")
        continue

    print(f"正在處理訓練檔案：{source_file}")

    training_data = pd.read_csv(source_file)
    training_data['DateTime'] = pd.to_datetime(training_data['DateTime'])
    merged_data = pd.merge(training_data, sun_data, on='DateTime', how='left')

    merged_data = merged_data.dropna(subset=['Azimuth', 'Elevation'])

    target_file = os.path.join(training_target_folder, f'L{i}.csv')
    merged_data.to_csv(target_file, index=False)

test_source_file = os.path.join(testing_source_folder, 'test_with_hualian.csv')
if os.path.exists(test_source_file):
    print(f"正在處理測試檔案：{test_source_file}")

    testing_data = pd.read_csv(test_source_file)
    testing_data['DateTime'] = pd.to_datetime(testing_data['DateTime'])
    merged_test_data = pd.merge(testing_data, sun_data, on='DateTime', how='left')

    merged_test_data = merged_test_data.dropna(subset=['Azimuth', 'Elevation'])

    test_target_file = os.path.join(testing_target_folder, 'test_with_hualian_SunAngle.csv')
    merged_test_data.to_csv(test_target_file, index=False)
else:
    print(f"測試檔案不存在：{test_source_file}")

print("所有檔案已成功處理並保存")
