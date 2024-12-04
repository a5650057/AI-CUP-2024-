import os
import pandas as pd
import re

location_story_map = {
    1: 5,
    2: 5,
    3: 5,
    4: 5,
    5: 5,
    6: 5,
    7: 5,
    8: 3,
    9: 3,
    10: 1,
    11: 1,
    12: 1,
    13: 5,
    14: 5,
    15: 1,
    16: 1,
    17: 1
}

folder_path = './'  

file_pattern = re.compile(r'^L([1-9]|1[0-7])__angle\.csv$')

for filename in os.listdir(folder_path):
    if file_pattern.match(filename):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)

            if 'LocationCode' not in df.columns:
                print(f"檔案 {filename} 中未找到 'LocationCode' 欄位，跳過。")
                continue

            df['story'] = df['LocationCode'].map(location_story_map)

            if df['story'].isnull().any():
                print(f"檔案 {filename} 中有無對應的 LocationCode，請檢查資料。")
                df['story'].fillna(0, inplace=True) 

            new_filename = filename.replace('_angle.csv', '_angle_story.csv')
            new_file_path = os.path.join(folder_path, new_filename)

            df.to_csv(new_file_path, index=False)
            print(f"已處理並儲存檔案: {new_filename}")

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")
