#Print most expensive car’s company name and price.

import pandas as pd
import os

df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Automobile_data.csv")
df = df [['company','price']][df.price==df['price'].max()]

print(df)
