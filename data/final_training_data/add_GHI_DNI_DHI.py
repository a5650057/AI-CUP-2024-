import os
import pandas as pd

pseudo_dir = "./"  
solar_data_path = "../../solar_irradiance_data_clear_sky_1min.csv"

solar_data = pd.read_csv(solar_data_path, parse_dates=["DateTime"])

for file_name in os.listdir(pseudo_dir):
    if file_name.endswith("story.csv"):
        
        file_path = os.path.join(pseudo_dir, file_name)
        pseudo_data = pd.read_csv(file_path, parse_dates=["DateTime"])
        
        merged_data = pd.merge(
            pseudo_data,
            solar_data[["DateTime", "DNI", "DHI", "GHI"]],
            on="DateTime",
            how="left"
        )
        
        merged_data.to_csv(file_path, index=False)
        print(f"已處理檔案: {file_name}")

print("所有檔案已更新完畢！")
