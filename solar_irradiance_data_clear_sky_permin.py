import pandas as pd

data = pd.read_csv('solar_irradiance_data_clear_sky_10min.csv')

data['DateTime'] = pd.to_datetime(data['DateTime'])

data.set_index('DateTime', inplace=True)

data_resampled = data.resample('1T').interpolate(method='linear').round(2)

data_resampled.reset_index(inplace=True)

data_resampled.to_csv('solar_irradiance_data_clear_sky_1min.csv', index=False)

print("success")
