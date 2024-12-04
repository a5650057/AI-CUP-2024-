import pandas as pd
from datetime import datetime, timedelta

def parse_serial_number(serial):
    date_part = serial[:8]
    time_part = serial[8:12]
    location_code = int(serial[12:14])
    
    datetime_str = f"{date_part} {time_part[:2]}:{time_part[2:]}:00"
    start_datetime = datetime.strptime(datetime_str, "%Y%m%d %H:%M:%S")
    end_datetime = start_datetime + timedelta(minutes=9)  
    return start_datetime, end_datetime, location_code

def calculate_average_power(test_df, start_dt, end_dt, location_code):
    mask = (
        (test_df['LocationCode'] == location_code) &
        (test_df['DateTime'] >= start_dt) &
        (test_df['DateTime'] <= end_dt)
    )
    filtered = test_df.loc[mask]
    
    if not filtered.empty:
        average_power = filtered['Predicted_Power(mW)'].mean()
        if average_power < 0:
            average_power = 0
        return round(average_power, 2)
    else:
        return None 
def main():
    
    test_predictions_path = 'test_predictions.csv'
    test_df = pd.read_csv(test_predictions_path)
    
    
    test_df['DateTime'] = pd.to_datetime(test_df['DateTime'])
    
   
    upload_path = '../../../../data/final_testing_data/upload(no answer).csv'
    upload_df = pd.read_csv(upload_path, dtype={'序號': str})  
    
    
    upload_df['答案'] = ''
    
   
    for index, row in upload_df.iterrows():
        serial = row['序號']
        start_dt, end_dt, location_code = parse_serial_number(serial)
        
        average_power = calculate_average_power(test_df, start_dt, end_dt, location_code)
        
        if average_power is not None:
            upload_df.at[index, '答案'] = average_power
        else:
            upload_df.at[index, '答案'] = 'N/A' 
    
  
    output_path = 'upload_with_answers.csv'
    upload_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"處理完成！結果已保存到 {output_path}")

if __name__ == "__main__":
    main()
