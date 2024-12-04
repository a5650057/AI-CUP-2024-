import pandas as pd
import os
from tqdm import tqdm


def apply_pseudo_labels():
    
    pseudo_labels_dir = './'
    final_test_file = os.path.join(pseudo_labels_dir, 'test_best.csv')
    training_data_dir = '../final_training_data'
    output_dir = pseudo_labels_dir  

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    print("讀取偽標籤文件...")
    try:
        df_pseudo = pd.read_csv(final_test_file, parse_dates=['DateTime'])
    except Exception as e:
        print(f"讀取偽標籤文件時出錯: {e}")
        return

    
    print(f"偽標籤數據總行數: {len(df_pseudo)}")
    print(f"偽標籤數據中的唯一 LocationCode 數量: {df_pseudo['LocationCode'].nunique()}")

   
    location_codes = df_pseudo['LocationCode'].unique()

    
    for loc in tqdm(location_codes, desc="處理各個LocationCode"):
        print(f'\n處理 LocationCode: {loc}')

        
        training_file = os.path.join(training_data_dir, f'L{loc}_angle_story.csv')

        
        if not os.path.exists(training_file):
            print(f'警告: {training_file} 不存在，跳過。')
            continue

        
        try:
            df_train = pd.read_csv(training_file, parse_dates=['DateTime'])
        except Exception as e:
            print(f"讀取訓練文件 {training_file} 時出錯: {e}")
            continue

        print(f"訓練文件 {training_file} 的總行數: {len(df_train)}")

       
        df_pseudo_loc = df_pseudo[df_pseudo['LocationCode'] == loc].copy()
        print(f"偽標籤數據中 LocationCode {loc} 的行數: {len(df_pseudo_loc)}")

        if len(df_pseudo_loc) == 0:
            print(f"警告: 沒有偽標籤數據在 LocationCode {loc}，跳過。")
            continue

        
        df_pseudo_loc.rename(columns={'Predicted_Power(mW)': 'Power(mW)'}, inplace=True)

       
        df_pseudo_loc['H_UVI_TEMP_interaction'] = ((df_pseudo_loc['H_UVI'] +0.01)* df_pseudo_loc['TEMP']).round(2)

       
        required_columns = ['LocationCode', 'DateTime', 'Power(mW)', 'H_UVI', 'TEMP', 'H_UVI_TEMP_interaction', 'GHI','Azimuth','Elevation','Solarinsolation','story','angle','DHI','DNI']
        missing_columns = [col for col in required_columns if col not in df_pseudo_loc.columns]
        if missing_columns:
            print(f"警告: 偽標籤數據中缺少欄位 {missing_columns}，跳過。")
            continue

        
        df_pseudo_loc = df_pseudo_loc[required_columns]

        
        duplicates = df_pseudo_loc.duplicated(subset=['DateTime']).sum()
        if duplicates > 0:
            print(f"警告: 偽標籤數據中有 {duplicates} 行重複的 DateTime，將移除重複。")
            df_pseudo_loc = df_pseudo_loc.drop_duplicates(subset=['DateTime'])

       
        existing_datetimes = set(df_train['DateTime'])
        df_pseudo_loc = df_pseudo_loc[~df_pseudo_loc['DateTime'].isin(existing_datetimes)]

        
        duplicates_after = df_pseudo_loc.duplicated(subset=['DateTime']).sum()
        if duplicates_after > 0:
            print(f"警告: 偽標籤數據中有 {duplicates_after} 行重複的 DateTime，將移除重複。")
            df_pseudo_loc = df_pseudo_loc.drop_duplicates(subset=['DateTime'])

        
        if len(df_pseudo_loc) == 0:
            print(f"警告: 偽標籤數據中沒有新的 DateTime 要追加到訓練文件 {training_file}，跳過。")
            continue

        
        df_train_updated = pd.concat([df_train, df_pseudo_loc], ignore_index=True)

        
        df_train_updated.sort_values(by='DateTime', inplace=True)

       
        df_train_updated['Power(mW)'] = df_train_updated['Power(mW)'].round(2)

        
        output_file = os.path.join(output_dir, f'L{loc}_Train_pseudo.csv')

        
        try:
            df_train_updated.to_csv(output_file, index=False)
            print(f"已保存更新後的文件到 {output_file}。總行數: {len(df_train_updated)}")
        except Exception as e:
            print(f"保存更新後的文件 {output_file} 時出錯: {e}")

    print('\n所有文件處理完成。')

if __name__ == "__main__":
    apply_pseudo_labels()
