import pandas as pd
import os
import json
import argparse

def prepare_solar_insolation(file_path):
    solar_df = pd.read_csv(
        file_path,
        parse_dates=['DateTime']
    )
    
    solar_df.set_index('DateTime', inplace=True)
    
    solar_df.sort_index(inplace=True)
    
    solar_min_df = solar_df.resample('T').ffill()
    
    solar_min_df.reset_index(inplace=True)
    
    return solar_min_df

def process_test_data(solar_min_df, customize=False):
    if customize:
        test_input_path = os.path.join('data', 'XGBoost_pseudo_labels_Data', 'final_test_interaction_with_GHI_SunAngle.csv')
        test_output_path = os.path.join('data', 'XGBoost_pseudo_labels_Data', 'final_test_interaction_with_GHI_SunAngle_Solarinsolation.csv')
        deleted_rows_json = os.path.join('data', 'XGBoost_pseudo_labels_Data', 'deleted_rows_test.json')
    else:
        test_input_path = os.path.join('data', 'final_testing_data', 'test_with_hualian_SunAngle.csv')
        test_output_path = os.path.join('data', 'final_testing_data', 'test_with_hualian_SunAngle_insolation.csv')
        deleted_rows_json = os.path.join('data', 'final_testing_data', 'deleted_rows_test.json')
    
    if not os.path.isfile(test_input_path):
        print(f"警告：測試數據文件 '{test_input_path}' 不存在，跳過。")
        return
    
    test_df = pd.read_csv(
        test_input_path,
        parse_dates=['DateTime']
    )
    
    merged_test_df = pd.merge(
        test_df,
        solar_min_df,
        on='DateTime',
        how='left'
    )
    
    missing_condition = merged_test_df[['Azimuth', 'Elevation', 'Solarinsolation']].isnull().any(axis=1)
    deleted_test_rows = merged_test_df[missing_condition][['LocationCode', 'DateTime']].copy()
    
    deleted_test_records = deleted_test_rows.copy()
    deleted_test_records['DateTime'] = deleted_test_records['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    deleted_test_records = deleted_test_records.to_dict(orient='records')
    
    with open(deleted_rows_json, 'w', encoding='utf-8') as json_file:
        json.dump(deleted_test_records, json_file, ensure_ascii=False, indent=4)
    
    cleaned_test_df = merged_test_df[~missing_condition].copy()
    
    cleaned_test_df.to_csv(test_output_path, index=False)
    
    print(f"成功將 Solarinsolation 數據添加到測試數據並保存為 '{test_output_path}'。")
    print(f"已刪除 {len(deleted_test_records)} 行缺失資料，詳細信息已保存至 '{deleted_rows_json}'。")

def process_training_data(solar_min_df, customize=False):
    if customize:
        print("在自訂模式下，不處理訓練數據。")
        return
    else:
        training_input_dir = os.path.join('data', 'final_training_data')
        training_output_dir = os.path.join('data', 'final_training_data')
        deleted_rows_json_path = os.path.join(training_output_dir, 'deleted_rows_training.json')
        
        os.makedirs(training_output_dir, exist_ok=True)
        
        all_deleted_records = [] 
        
       
        for i in range(1, 18):
            input_filename = f'L{i}_.csv'
            input_path = os.path.join(training_input_dir, input_filename)
            
            if not os.path.isfile(input_path):
                print(f"警告：文件 '{input_path}' 不存在，跳過。")
                continue
            
            train_df = pd.read_csv(
                input_path,
                parse_dates=['DateTime']
            )
            
            merged_train_df = pd.merge(
                train_df,
                solar_min_df,
                on='DateTime',
                how='left'
            )
            
            missing_condition = merged_train_df[['Azimuth', 'Elevation', 'Solarinsolation']].isnull().any(axis=1)
            deleted_train_rows = merged_train_df[missing_condition][['LocationCode', 'DateTime']].copy()
            
            deleted_train_records = deleted_train_rows.copy()
            deleted_train_records['DateTime'] = deleted_train_records['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            deleted_train_records = deleted_train_records.to_dict(orient='records')
            
            all_deleted_records.extend(deleted_train_records)
            
            cleaned_train_df = merged_train_df[~missing_condition].copy()
            
            output_filename = f'L{i}_.csv'
            output_path = os.path.join(training_output_dir, output_filename)
            
            cleaned_train_df.to_csv(output_path, index=False)
            
            print(f"成功將 Solarinsolation 數據添加到訓練數據 '{input_filename}' 並保存為 '{output_path}'。")
            print(f"已刪除 {len(deleted_train_records)} 行缺失資料。")
        
        if all_deleted_records:
            with open(deleted_rows_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(all_deleted_records, json_file, ensure_ascii=False, indent=4)
            print(f"所有訓練數據中已刪除的行詳細信息已保存至 '{deleted_rows_json_path}'。")
        else:
            print("訓練數據中沒有缺失的行需要刪除。")

def main():
    parser = argparse.ArgumentParser(description="處理 Solarinsolation 數據並合併到測試和訓練數據中。")
    parser.add_argument('--customize', action='store_true', help='使用自訂的測試數據路徑。')
    
    args = parser.parse_args()
    customize = args.customize
    
    solar_insolation_path = 'Solar_Insolation_10min.csv'
    
    if not os.path.isfile(solar_insolation_path):
        print(f"錯誤：文件 '{solar_insolation_path}' 不存在。請確保文件位於腳本目錄中。")
        return
    
    solar_min_df = prepare_solar_insolation(solar_insolation_path)
    print("成功將 Solar_Insolation_10min.csv 轉換為每分鐘的 Solarinsolation 數據。")
    
    process_test_data(solar_min_df, customize)
    
    process_training_data(solar_min_df, customize)
    
    print("所有文件已成功處理並保存。")

if __name__ == "__main__":
    main()
