使用python 虛擬環境  Python 3.10.13

建立虛擬環境 : python -m venv myvenv

啟動虛擬環境 : .\myvenv\Scripts\activate

安裝套件 : pip install -r requirements.txt

安裝CUDA : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



資料前處理:

(由於需要用到openweathermap APIKEY 故直接把處理完的資料放到data裡 以下只是步驟說明)

1. 執行 create_final_traingdata.py
把第二階段的訓練集與第一階段結合
結合後的訓練集會直接在原先第一階段的資料夾內做更新 : data/original_data

2. 執行 remove_duplicates.py

檢查訓練集中 重複的時間 
如有重複 則保留第一筆資料 其他的移除
並輸出到remove_duplicate資料夾中

3. 將 upload(no answer).csv 放到data/final_testing_data

執行 expandTestData.py 

此腳本會將原先時間區間從10分鐘 改為1分鐘 且格式化成datetime 形式

輸出: expanded_upload.csv


4. 執行 add_HuaLiandata_to_data.py

此腳本會把花蓮氣溫與紫外線 以及此兩個特徵的相乘數值加入到訓練集 
輸出LX_Train_with_hualianData.csv

接著執行 add_HuaLiandata_to_data.py --testdata
處理測試集 輸出: test_with_hualian.csv

接著直接開啟L1_Train_with_hualianData.csv  與  L17_Train_with_hualianData.csv
把1月1號 八點之前的時間點刪掉 (因為這些時間 我們沒有花蓮天氣資料)


4. 

執行 solar_data_creater.py  

此腳本會使用openweathermap的API獲取晴空天氣下的GHI DHI  DNI

輸出: solar_irradiance_data_clear_sky.csv

5. 接著依序執行 
solar_irradiance_data_clear_sky_per10min.py
solar_irradiance_data_clear_sky_permin.py

把原先15分鐘的資料 經過線性插值 變成美分鐘的資料
最終會輸出: solar_irradiance_data_clear_sky_1min.csv

6. 
執行
sun_Azimuth_elevation_angle_generator.py

用pvlib計算出某經緯度的太陽高度角 與 太陽方位角
輸出: sun_Azimuth_elevation_angle.csv

接著執行

sun_Azimuth_elevation_angle_addtoData.py
把相對時間的太陽高度角 方位角 添加到訓練集與測試集中


7. 執行 Solar_Insolation_ LinearInterpolation.py
將每小時太陽日射量 進行線性插值 變成每10分鐘一筆

輸出: Solar_Insolation_10min.csv


8. 執行 add_Solar_Insolation_toDATA.py
此腳本會將太陽日射量插入到訓練集與測試集中


9.  
移動到 data/final_training_data 資料夾
依序執行

add_angle.py
add_story.py
add_GHI_DNI_DHI.py

以及data/final_testing_data
依序執行

add_angle.py
add_story.py
add_GHI_DNI_DHI.py



以上皆為資料處理部分


接著是重現模型訓練的部分:



