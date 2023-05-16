#Python pandas  first 5 rows
import pandas as pd
import os

print(os.path.dirname(os.path.realpath(__file__)))
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Automobile_data.csv")
print(df.head(5))

#Python pandas  last 5 rows
print("Last 5 rows")
print(df.tail(5))
