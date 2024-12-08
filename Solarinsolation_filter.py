import pandas as pd
import csv

file_path = "72T250_item_hour_20241208234803.csv"

try:
    data = pd.read_csv(file_path, na_values=["", " "], on_bad_lines='skip')
except TypeError:
    data = pd.read_csv(file_path, na_values=["", " "], error_bad_lines=False)

data['Solarinsolation'].fillna(0, inplace=True)

data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
data = data.dropna(subset=['DateTime'])

start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2024-10-30 23:59:59")
data = data[(data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)]

data = data[data['DateTime'].dt.hour.between(6, 17)]

data = data.set_index('DateTime')

data = data[~data.index.duplicated(keep='first')]

unique_dates = sorted(pd.Series(data.index.date).unique())

all_days_complete = []
for d in unique_dates:
    full_range = pd.date_range(start=pd.Timestamp(d).replace(hour=6, minute=0),
                               end=pd.Timestamp(d).replace(hour=19, minute=0),
                               freq='H')

    daily_data = data[data.index.date == d]

    daily_data = daily_data.reindex(full_range)

    daily_data['Solarinsolation'] = daily_data['Solarinsolation'].fillna(0)

    
    all_days_complete.append(daily_data)

complete_data = pd.concat(all_days_complete)

complete_data['Solarinsolation'] = complete_data['Solarinsolation'].astype(float).apply(lambda x: f"{x:.2f}")

complete_data = complete_data.reset_index()
complete_data.rename(columns={'index': 'DateTime'}, inplace=True)
complete_data['DateTime'] = complete_data['DateTime'].dt.strftime('%Y/%m/%d %H:%M')


final_data = complete_data[['DateTime', 'Solarinsolation']]

final_data.sort_values(by='DateTime', inplace=True)

output_path = "Solar_Insolation.csv"
final_data.to_csv(output_path, index=False)

print(f"過濾後且補齊每個時段的資料已儲存到 {output_path}")
