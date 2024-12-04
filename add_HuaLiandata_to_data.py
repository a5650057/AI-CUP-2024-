import pandas as pd
import numpy as np
import os
import glob
import argparse

def add_UVI_TEMP_to_files(filled_csv_path, input_files_dir, output_dir, is_test=False):

    if not os.path.isfile(filled_csv_path):
        print(f"花蓮_final.csv 檔案未找到: {filled_csv_path}")
        return

    if not os.path.isdir(input_files_dir):
        print(f"輸入檔案目錄未找到: {input_files_dir}")
        return

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"已建立輸出目錄: {output_dir}")

    try:
        df_filled = pd.read_csv(filled_csv_path, parse_dates=['time'])
    except Exception as e:
        print(f"讀取花蓮_final.csv時發生錯誤: {e}")
        return

    if 'time' not in df_filled.columns:
        print("花蓮_final.csv 中找不到 'time' 欄位。")
        return

    df_filled = df_filled.sort_values('time').reset_index(drop=True)

    if is_test:
        test_file = os.path.join(input_files_dir, 'expanded_upload.csv')
        if not os.path.isfile(test_file):
            print(f"警告: 測試檔案未找到，跳過: {test_file}")
            return

        print(f"處理測試檔案 {os.path.basename(test_file)}...")

        try:
            df_test = pd.read_csv(test_file, parse_dates=['DateTime'])
        except Exception as e:
            print(f"讀取測試檔案 {test_file} 時發生錯誤: {e}")
            return

        required_columns = ['DateTime']
        if not all(col in df_test.columns for col in required_columns):
            print(f"測試檔案 {os.path.basename(test_file)} 缺少必要的欄位，跳過。")
            return

        df_test = df_test.sort_values('DateTime').reset_index(drop=True)

        df_test = pd.merge_asof(
            df_test.sort_values('DateTime'),
            df_filled[['time', 'H_UVI', 'TEMP']].sort_values('time'),
            left_on='DateTime',
            right_on='time',
            direction='backward',
            tolerance=pd.Timedelta('10min')
        )

        df_test = df_test.drop(columns=['time'])
        df_test['H_UVI'] = df_test['H_UVI'].round(1)

        unmatched = df_test[df_test['H_UVI'].isna() | df_test['TEMP'].isna()]
        if not unmatched.empty:
            print(f"警告: 在檔案 {os.path.basename(test_file)} 中有 {len(unmatched)} 筆資料未匹配到 H_UVI、TEMP。")
            df_test['H_UVI'] = df_test['H_UVI'].round(1)
            df_test['TEMP'] = df_test['TEMP'].ffill()

        df_test['H_UVI_TEMP_interaction'] = ((df_test['H_UVI']+0.01) * df_test['TEMP']).round(2)

       

        output_filename = 'test_with_hualian.csv'
        output_path = os.path.join(output_dir, output_filename)

        df_test.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已儲存更新後的測試檔案: {output_filename}")

    else:
        train_file_pattern = os.path.join(input_files_dir, 'L*_Train_deduplicated.csv')
        train_files = glob.glob(train_file_pattern)

        if not train_files:
            print(f"在目錄 {input_files_dir} 中找不到符合模式的訓練檔案。")
            return

        print(f"找到 {len(train_files)} 個訓練檔案。")

        for train_file in train_files:
            try:
                base_filename = os.path.basename(train_file)
                name_parts = base_filename.split('_')
                if len(name_parts) < 2:
                    print(f"檔案名稱格式不正確，跳過: {base_filename}")
                    continue
                i_part = name_parts[0][1:]
                i = i_part

                print(f"處理檔案 {base_filename}...")

                df_train = pd.read_csv(train_file, parse_dates=['DateTime'])

                required_columns = ['DateTime']
                if not all(col in df_train.columns for col in required_columns):
                    print(f"檔案 {base_filename} 缺少必要的欄位，跳過。")
                    continue

                df_train = df_train.sort_values('DateTime').reset_index(drop=True)

                df_train = pd.merge_asof(
                    df_train.sort_values('DateTime'),
                    df_filled[['time', 'H_UVI', 'TEMP']].sort_values('time'),
                    left_on='DateTime',
                    right_on='time',
                    direction='backward',
                    tolerance=pd.Timedelta('10min')
                )

                df_train = df_train.drop(columns=['time'])
                df_train['H_UVI'] = df_train['H_UVI'].round(1)
                unmatched = df_train[df_train['H_UVI'].isna() | df_train['TEMP'].isna()]
                if not unmatched.empty:
                    print(f"警告: 在檔案 {base_filename} 中有 {len(unmatched)} 筆資料未匹配到 H_UVI、TEMP")
                    df_train['H_UVI'] = df_train['H_UVI'].round(1)
                    df_train['TEMP'] = df_train['TEMP'].ffill()

                df_train['H_UVI_TEMP_interaction'] = ((df_train['H_UVI']+0.01) * df_train['TEMP']).round(2)

                output_filename = f"L{i}_Train_with_hualianData.csv"
                output_path = os.path.join(output_dir, output_filename)

                df_train.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"已儲存更新後的檔案: {output_filename}")

            except Exception as e:
                print(f"處理檔案 {train_file} 時發生錯誤: {e}")
                continue

        print("所有訓練檔案處理完成。")

def main():
    parser = argparse.ArgumentParser(description="將csv中的H_UVI、TEMP欄位加入到Train或Test CSV檔案中，並計算H_UVI_TEMP_interaction。")
    parser.add_argument('--testdata', action='store_true', help='處理測試資料')

    args = parser.parse_args()

    filled_csv = '花蓮_final.csv'

    if args.testdata:
        input_files_directory = os.path.join('data', 'final_testing_data')
        output_directory = os.path.join('data', 'final_testing_data')
        add_UVI_TEMP_to_files(filled_csv, input_files_directory, output_directory, is_test=True)
    else:
        input_files_directory = os.path.join('data', 'remove_duplicate')
        output_directory = os.path.join('data', 'remove_duplicate')
        add_UVI_TEMP_to_files(filled_csv, input_files_directory, output_directory, is_test=False)

if __name__ == "__main__":
    main()
