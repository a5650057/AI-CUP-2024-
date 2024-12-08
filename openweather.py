import requests
import csv
import schedule
import time
from datetime import datetime
import os


def get_weather_data(auth_key, station_id="466990", response_format="JSON"):
    """
    Fetch weather data for the specified station.

    :param auth_key: Authorization key for the weather open data platform
    :param station_id: Station ID, default is "466990"
    :param response_format: Data format, default is "JSON"
    :return: JSON data response
    """
    base_url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001"

    params = {
        "Authorization": auth_key,
        "StationId": station_id,
        "format": response_format,
        "WeatherElement": "UVIndex,AirTemperature" 
    }

    try:
        print(f"[{datetime.now()}] Sending request to: {base_url} with params: {params}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        print("Data fetched successfully.")
        return data
    except requests.exceptions.RequestException as req_err:
        print(f"[{datetime.now()}] Request exception: {req_err}")
    except Exception as err:
        print(f"[{datetime.now()}] Other error occurred: {err}")


def load_last_timestamp(filename):
    """
    Load the last observation time from the existing CSV file.

    :param filename: CSV filename (including path)
    :return: The last observation time as a string, or None if the file doesn't exist or is empty
    """
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                last_row = None
                for last_row in reader:
                    pass  
                if last_row:
                    return last_row["time"]
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
    return None


def save_json_to_csv(data, filename):
    """
    Save JSON data to the specified CSV file.

    :param data: JSON data to save
    :param filename: Target CSV filename (including path)
    """
    try:
        
        headers = ["time", "H_UVI", "TEMP"]

       
        last_timestamp = load_last_timestamp(filename)

       
        with open(filename, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            
            if csvfile.tell() == 0:
                writer.writeheader()

           
            for station in data.get("records", {}).get("Station", []):
               
                obs_time_raw = station.get("ObsTime", {}).get("DateTime", "")
                if obs_time_raw:
                    try:
                        dt = datetime.fromisoformat(obs_time_raw)
                        obs_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        obs_time = obs_time_raw.replace("T", " ").split("+")[0]
                else:
                    obs_time = ""

                
                if obs_time == last_timestamp:
                    print(f"[{datetime.now()}] Time {obs_time} is already recorded, skipping.")
                    continue

             
                element_dict = station.get("WeatherElement", {})
                H_UVI = element_dict.get("UVIndex", "0.0")
                TEMP = element_dict.get("AirTemperature", "0.0")

                try:
                    H_UVI = f"{float(H_UVI):.2f}"  
                    TEMP = f"{float(TEMP):.2f}"   
                except ValueError:
                    print(f"Invalid number format in H_UVI or TEMP: H_UVI={H_UVI}, TEMP={TEMP}")

                
                row = {
                    "time": obs_time,
                    "H_UVI": H_UVI,
                    "TEMP": TEMP
                }

               
                writer.writerow(row)
                print(f"[{datetime.now()}] Wrote time {obs_time} to CSV.")

        print(f"CSV data has been successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving CSV data: {e}")


def job():
    AUTHORIZATION_KEY = "CWA-A6D73978-FC8C-490B-B3AE-A12A4DB34274"  
    weather_data = get_weather_data(auth_key=AUTHORIZATION_KEY)
    if weather_data:
        filename = "weather_data_hualian.csv"
        save_json_to_csv(weather_data, filename)


if __name__ == "__main__":
    schedule.every(1).minutes.do(job)
    print("Starting scheduled task to fetch and save weather data every minute. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)
