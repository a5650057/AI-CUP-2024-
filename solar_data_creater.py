import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os

API_KEY_FILE = 'api_key.json' 
LATITUDE = 23.899444
LONGITUDE = 121.544444
START_DATE = '2024-01-01'
END_DATE = '2024-10-30'
INTERVAL = '15m'  
TIMEZONE = '+08:00'  


OUTPUT_CSV = 'solar_irradiance_data_clear_sky.csv'


API_URL = 'https://api.openweathermap.org/energy/1.0/solar/interval_data'

def load_api_key(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"API密鑰檔案 '{file_path}' 不存在。請確保該檔案存在並包含正確的API密鑰。")
    
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            api_key = data.get('API_KEY')
            if not api_key:
                raise KeyError("JSON檔案中未找到 'API_KEY' 鍵。請確保JSON檔案包含正確的鍵。")
            return api_key
        except json.JSONDecodeError:
            raise ValueError("API密鑰檔案格式錯誤。請確保檔案是有效的JSON格式。")
        
def generate_date_range(start_date, end_date):
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end - start
    date_list = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return [date.strftime('%Y-%m-%d') for date in date_list]

def fetch_solar_data(date, lat, lon, interval, tz, api_key):
   
    params = {
        'lat': lat,
        'lon': lon,
        'date': date,
        'interval': interval,
        'tz': tz,
        'appid': api_key
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for date {date}: {http_err}")
    except Exception as err:
        print(f"Other error occurred for date {date}: {err}")
    return None

def extract_clear_sky_data(api_response, date):
    records = []
    try:
        intervals = api_response['irradiance']['intervals']
        for interval in intervals:
            start_time = interval['start']
            
            if '05:00' <= start_time < '19:00':
                datetime_str = f"{date} {start_time}"
                ghi = interval['clear_sky']['ghi']
                dni = interval['clear_sky']['dni']
                dhi = interval['clear_sky']['dhi']
                records.append({
                    'DateTime': datetime_str,
                    'GHI': ghi,
                    'DNI': dni,
                    'DHI': dhi
                })
    except KeyError as e:
        print(f"Key error when extracting data for date {date}: {e}")
    return records

def main():
   
    try:
        API_KEY = load_api_key(API_KEY_FILE)
        print("成功載入API密鑰。")
    except Exception as e:
        print(f"載入API密鑰時出錯: {e}")
        return
    
  
    dates = generate_date_range(START_DATE, END_DATE)
    print(f"Total dates to process: {len(dates)}")
    
    all_records = []
    
    for idx, date in enumerate(dates, 1):
        print(f"Processing {idx}/{len(dates)}: {date}")
        api_response = fetch_solar_data(date, LATITUDE, LONGITUDE, INTERVAL, TIMEZONE, API_KEY)
        if api_response:
            records = extract_clear_sky_data(api_response, date)
            if records:
                all_records.extend(records)
            else:
                print(f"No data extracted for date {date}.")
        else:
            print(f"Failed to retrieve data for date {date}. Skipping.")
        
        
        time.sleep(1) 
    
   
    if not all_records:
        return
    
   
    df = pd.DataFrame(all_records)
    
 
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Data successfully saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()