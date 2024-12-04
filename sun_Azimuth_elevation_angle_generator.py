import pandas as pd
import pvlib
from pvlib.location import Location

latitude = 23.975128
longitude = 121.613275
tz = 'Asia/Taipei' 

site = Location(latitude, longitude, tz=tz)

start_date = '2024-01-01'
end_date = '2024-10-31'

date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz=tz)

results = []

for single_date in date_range:
    times = pd.date_range(
        start=single_date + pd.Timedelta(hours=5),
        end=single_date + pd.Timedelta(hours=19),
        freq='1min',
        tz=tz
    )
    
    solar_position = site.get_solarposition(times)
    
    data = pd.DataFrame({
        'DateTime': times.strftime('%Y-%m-%d %H:%M:%S'),
        'Azimuth': solar_position['azimuth'].round(2),
        'Elevation': solar_position['apparent_elevation'].round(2),
    })
    
    data = data[data['Elevation'] >= 0]
    
    results.append(data)

final_data = pd.concat(results, ignore_index=True)

final_data.to_csv('sun_Azimuth_elevation_angle.csv', index=False)

print("已保存到 'sun_Azimuth_elevation_angle.csv' 文件")
