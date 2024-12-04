import pandas as pd
from datetime import datetime, timedelta

input_file = 'upload(no answer).csv'
output_file = 'expanded_upload.csv'

df = pd.read_csv(input_file, dtype={'序號': str})

expanded_rows = []

for index, row in df.iterrows():
    serial_number = row['序號']
    
    if pd.isnull(serial_number) or serial_number.strip() == '':
        continue  
    
    if '.' in serial_number:
        serial_number = serial_number.split('.')[0]

    location_code_str = serial_number[-2:]
    try:
        location_code = int(location_code_str) 
    except ValueError:
        print(f"序號 {serial_number} 的 LocationCode 無法轉換為整數，跳過該序號。")
        continue


    datetime_str = serial_number[:12]


    try:
        start_datetime = datetime.strptime(datetime_str, '%Y%m%d%H%M')
    except ValueError:
        print(f"序號 {serial_number} 的日期時間格式不正確，跳過該序號。")
        continue

    for i in range(10):
        current_datetime = start_datetime + timedelta(minutes=i)
        expanded_rows.append({
            'LocationCode': location_code,
            'DateTime': current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'Power(mW)': None  
        })


expanded_df = pd.DataFrame(expanded_rows)

expanded_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"已成功將資料擴展並輸出到 {output_file}")
