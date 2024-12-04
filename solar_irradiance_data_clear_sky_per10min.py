import pandas as pd


input_file = "solar_irradiance_data_clear_sky.csv"  
output_file = "solar_irradiance_data_clear_sky_10min.csv"

df = pd.read_csv(input_file)
df['DateTime'] = pd.to_datetime(df['DateTime'])

new_time_range = pd.date_range(
    start=df['DateTime'].min(),
    end=df['DateTime'].max(),
    freq='10T'  
)

df.set_index('DateTime', inplace=True)

interpolated_df = df.reindex(new_time_range).interpolate(method='linear')

interpolated_df = interpolated_df.round(2)

interpolated_df.reset_index(inplace=True)
interpolated_df.rename(columns={'index': 'DateTime'}, inplace=True)

filtered_df = interpolated_df[
    (interpolated_df['DateTime'].dt.time >= pd.Timestamp("05:00:00").time()) & 
    (interpolated_df['DateTime'].dt.time <= pd.Timestamp("19:00:00").time())
]

filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"線性插值完成，結果已儲存至 {output_file}")