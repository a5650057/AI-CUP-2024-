import os
import pandas as pd

pseudo_dir = "./"  
solar_data_path = "../../solar_irradiance_data_clear_sky_1min.csv"

solar_data = pd.read_csv(solar_data_path, parse_dates=["DateTime"])

all_merged_data = pd.DataFrame()

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
        
        all_merged_data = pd.concat([all_merged_data, merged_data], ignore_index=True)
        print(f"已處理: {file_name}")

output_file_path = os.path.join(pseudo_dir, "final_test_angle_story.csv")

all_merged_data.to_csv(output_file_path, index=False)
print("final_test_angle_story.csv")
