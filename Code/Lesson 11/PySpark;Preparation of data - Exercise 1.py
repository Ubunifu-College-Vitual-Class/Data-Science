# Your task is to perform the following data analysis tasks using PySpark:
# Load the dataset into a PySpark DataFrame and inspect its structure.
# Calculate the average quantity purchased by the customers.
# Find the stockcode distribution among the customers.
# Calculate the total and average sales made.
# Calculate the total and average amount spent by the customers per stockcode/model.
# Identify the customers with the highest total spend.
# Calculate the correlation coefficient between country and total spend.
# Perform a group-by operation to find the average amount spent by customers based on their country.
# Create a new column called ‘spending_category' based on the amount spent by the customers. Group the customers into the following  groups: ‘Heavy spender' ( >= 30000), ‘Average spender' (30000 -15000), and ‘Low spender' ( < 15000). Calculate the average amount spent by customers in each spending group.
# Visualize the distribution of StockCode using a histogram.



import os
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName('Ubunifu_College_PySpark_Lesson').getOrCreate()

# Load the dataset into a PySpark DataFrame and inspect its structure.
csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\ECommerceStockList.csv"
data = spark.read.csv(
    csv_file,
    sep = ',',
    header = True
    )


print(data.count())



# Calculate the average quantity purchased by the customers.
from pyspark.sql import functions as f
average = data.select(f.avg(data["Quantity"])).first()[0]
print("Average quantity:", average)

# Find the stockcode distribution among the customers.
distribution = data.groupBy("StockCode").count()
distribution.show()

# Calculate the total and average sales made.
data = data.withColumn('Total', f.col('Quantity') * f.col('UnitPrice'))
total_sales = data.select(f.sum(data["Total"])).first()
average_sales = data.select(f.avg(data["Total"])).first()
print("Average sales:", average_sales)
print("Total sales:", total_sales)


#Calculate the total and average amount spent by the customers per stockcode/model.
totals_and_averages = data.groupBy("StockCode").agg(
    f.sum(data["Total"]).alias("total_spent_by_customer"),
    f.avg(data["Total"]).alias("average_spent_by_customer")
)
totals_and_averages.show()



#Identify the customers with the highest total spend.
highest_record_row = data.orderBy(f.desc("Total")).limit(1)
highest_record_row.show()



# Calculate the correlation coefficient between country and total spend.
correlation_coefficient = data.select(f.corr(data["Country"], data["Total"])).first()
print("Correlation Coefficient:", correlation_coefficient) # The correlation coefficient ranges between -1 and 1, where values close to 1 indicate a strong positive correlation, values close to -1 indicate a strong negative correlation, and values close to 0 indicate no or weak correlation.




# Perform a group-by operation to find the average amount spent by customers based on their country.
average_spent_by_country = data.groupBy("Country").agg(f.avg(data["Total"]).alias("Average_amount_spent"))
average_spent_by_country.show()



# Create a new column called ‘spending_category' based on the amount spent by the customers. Group the customers into the following  groups: ‘Heavy spender' ( >= 30000), ‘Average spender' (30000 -15000), and ‘Low spender' ( < 15000). Calculate the average amount spent by customers in each spending group.
data = data.withColumn("spending_category", 
                   f.when(data["Total"] >= 30000, "Heavy spender")
                   .when((data["Total"] >= 5000) & (data["Total"] < 20000), "Average spender")
                   .otherwise("Low spender"))
average_spent_by_category = data.groupBy("spending_category").agg(f.avg(data["Total"]).alias("Average_spent"))
average_spent_by_category.show()




# Visualize the distribution of annual income using a histogram.
import matplotlib.pyplot as plt
import seaborn as sns

data = data.withColumn("StockCode", data["StockCode"].cast("float"))
income_pandas = data.select("StockCode").toPandas()
plt.figure(figsize=(10, 6))
sns.histplot(income_pandas["StockCode"], bins=20, kde=True)
plt.xlabel("Stock Code")
plt.ylabel("Frequency")
plt.title("Distribution of Stock Code")
plt.show()




























































































































































































