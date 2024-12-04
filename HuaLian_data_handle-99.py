import pandas as pd
import numpy as np

def interpolate_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file, parse_dates=['time'])
        
        df.replace(-99.0, np.nan, inplace=True)
        
        numeric_columns = df.columns.drop('time')
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        df.set_index('time', inplace=True)
        
        df_interpolated = df.interpolate(method='linear')
        
        df_interpolated.reset_index(inplace=True)
        
        df_interpolated.to_csv(output_file, index=False, float_format='%.2f')
        print(f"插值完成，結果已儲存至 {output_file}")
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    input_csv = "花蓮_data.csv"        
    output_csv = "花蓮_final.csv"  

    interpolate_csv(input_csv, output_csv)
