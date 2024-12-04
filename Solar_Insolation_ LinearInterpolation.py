import pandas as pd


df = pd.read_csv('Solar_Insolation.csv', parse_dates=['DateTime'], date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M'))

df.set_index('DateTime', inplace=True)

df.sort_index(inplace=True)


df_resampled = df.resample('10T').interpolate(method='linear')


df_resampled['Solarinsolation'] = df_resampled['Solarinsolation'].round(2)

df_resampled.reset_index(inplace=True)

df_resampled.to_csv('Solar_Insolation_10min.csv', index=False)

print("成功將每小時的 Solarinsolation 數據轉換為每十分鐘的數據並保存到 'Solar_Insolation_10min.csv'。")
