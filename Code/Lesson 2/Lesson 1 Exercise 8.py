#Read toothpaste sales data of each month and show it using a scatter plot (Dataset company_sales_data.csv)

import pandas as pd
import matplotlib.pyplot as plt
import os   

df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\company_sales_data.csv")

monthList  = df ['month_number'].tolist()
toothPasteSalesData = df ['toothpaste'].tolist()

plt.scatter(monthList, toothPasteSalesData, label = 'Tooth paste Sales data')

plt.xlabel('Month Number')
plt.ylabel('Number of units Sold')
plt.legend(loc='upper left')

plt.title(' Tooth paste Sales data')
plt.xticks(monthList)

plt.grid(True, linewidth= 1, linestyle="--")

plt.show()
