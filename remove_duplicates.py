import os
import pandas as pd

data_folder = './data/original_data'  
output_folder = './data/remove_duplicate' 
os.makedirs(output_folder, exist_ok=True)  


duplicate_records = []

for i in range(1, 18):
    file_path = os.path.join(data_folder, f'L{i}_Train.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        df['DateTime'] = pd.to_datetime(df['DateTime']).dt.floor('T') 

        duplicates = df[df.duplicated(subset=['LocationCode', 'DateTime'], keep=False)]
        
        if not duplicates.empty:
            duplicate_records.append((file_path, duplicates))
        
        df_deduplicated = df.drop_duplicates(subset=['LocationCode', 'DateTime'], keep='first')


        if 'WindSpeed(m/s)' in df_deduplicated.columns:
            df_deduplicated = df_deduplicated.drop(columns=['WindSpeed(m/s)'])
            df_deduplicated = df_deduplicated.drop(columns=['Pressure(hpa)'])
            df_deduplicated = df_deduplicated.drop(columns=['Temperature(°C)'])
            df_deduplicated = df_deduplicated.drop(columns=['Humidity(%)'])
            df_deduplicated = df_deduplicated.drop(columns=['Sunlight(Lux)'])

        output_file_path = os.path.join(output_folder, f'L{i}_Train_deduplicated.csv')
        df_deduplicated.to_csv(output_file_path, index=False)
        
    else:
        print(f"文件 {file_path} 不存在")

for file_path, duplicates in duplicate_records:
    print(f"文件 {file_path} 中找到重複項:")
    print(duplicates)
