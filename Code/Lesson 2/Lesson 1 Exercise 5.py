#Clean the dataset and update the CSV file
import pandas as pd
import os

df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Automobile_data.csv", na_values={
'price':["?","n.a"],
'stroke':["?","n.a"],
'horsepower':["?","n.a"],
'peak-rpm':["?","n.a"],
'average-mileage':["?","n.a"]})
print (df)
df.to_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Automobile_data_cleaned.csv")
