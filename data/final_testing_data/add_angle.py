import os
import pandas as pd
location_angle_map = {
    1: 181,
    2: 175,
    3: 180,
    4: 161,
    5: 208,
    6: 208,
    7: 172,
    8: 219,
    9: 151,
    10: 223,
    11: 131,
    12: 298,
    13: 249,
    14: 197,
    15: 127,
    16: 82,
    17: 200
}

folder_path = './'  

for filename in os.listdir(folder_path):
    if filename.endswith('_insolation.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)

            if 'LocationCode' not in df.columns:
                print(f"檔案 {filename} 中未找到 'LocationCode' 欄位，跳過。")
                continue

            df['angle'] = df['LocationCode'].map(location_angle_map)

            if df['angle'].isnull().any():
                print(f"檔案 {filename} 中有無對應的 LocationCode，請檢查資料。")
                df['angle'].fillna(0, inplace=True)  

            new_filename = f"{os.path.splitext(filename)[0]}_angle.csv"
            new_file_path = os.path.join(folder_path, new_filename)

            df.to_csv(new_file_path, index=False)
            print(f"已處理並儲存檔案: {new_filename}")

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")
