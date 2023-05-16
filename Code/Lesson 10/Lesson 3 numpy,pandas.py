# Using the dataset in Kenya Tree Cover (.CSV)
# Link : https://africaopendata.org/dataset/kenyan-forests-datasets/resource/cbc08cec-4ff0-4c62-9147-705bf37baab1 

import pandas as pd
import os
#import mysql.connector #pip install mysql-connector-python
#from mysql.connector import Error 
from sqlalchemy import create_engine  #pip install mysqlclient

# 2). Import and create a pandas dataframe
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\KenyaTreeCover.csv")

# 3). Iterate through the dataframe #iterate means looping
for index, row in df.iterrows():
    print(row['lat'] ,row['lon']) # 4). From the iterated data frame display the latitude and longitude

print("##################")
# 5). Detele rows on the dataframe where the latitude is a negative number

df_filtered = df[df['lat'] > 0]
#display the result
for index, rowfiltered in df_filtered.iterrows():
    print(rowfiltered['lat']) 

print("##################")
# 6). Save you dataframe in mysql database
connection = create_engine("mysql+mysqldb://root:1234@localhost/python_sql_exercise") #fill details
df.to_sql(con=connection,name='data_frame_to_mysql',if_exists='append', index=False)

