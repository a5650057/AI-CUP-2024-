import os
import pandas as pd


main_data_dir = 'data/original_data'
additional_data_dir = 'data/addHualianData_additional'
output_dir = 'data/original_data'  

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

columns_to_keep = ['LocationCode', 'DateTime', 'Power(mW)']


main_files = [f for f in os.listdir(main_data_dir) if f.endswith('.csv')]

for main_filename in main_files:
    location_code = main_filename.split('_')[0]  
    
    main_file_path = os.path.join(main_data_dir, main_filename)
    
    additional_filename = f"{location_code}_Train_2.csv"
    additional_file_path = os.path.join(additional_data_dir, additional_filename)
    
    combined_df = pd.DataFrame()
    
    try:
        main_df = pd.read_csv(main_file_path, parse_dates=['DateTime'])
        combined_df = pd.concat([combined_df, main_df], ignore_index=True)
    except Exception as e:
        print(f"讀取主檔案 {main_filename} 時出錯: {e}")
        continue
    
    if os.path.exists(additional_file_path):
        try:
            additional_df = pd.read_csv(additional_file_path, parse_dates=['DateTime'])
            combined_df = pd.concat([combined_df, additional_df], ignore_index=True)
        except Exception as e:
            print(f"讀取附加檔案 {additional_filename} 時出錯: {e}")
            pass
    
    combined_df.drop_duplicates(subset=['LocationCode', 'DateTime'], inplace=True)
    
    available_columns = [col for col in columns_to_keep if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
    if 'DateTime' in combined_df.columns:
        combined_df['DateTime'] = pd.to_datetime(combined_df['DateTime'], errors='coerce')
        combined_df.dropna(subset=['DateTime'], inplace=True)
    
    if 'DateTime' in combined_df.columns:
        combined_df.sort_values(by='DateTime', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
    
    output_file_path = os.path.join(output_dir, main_filename)
    try:
        combined_df.to_csv(output_file_path, index=False)
        print(f"已成功處理並輸出檔案：{output_file_path}")
    except Exception as e:
        print(f"寫入檔案 {output_file_path} 時出錯: {e}")
