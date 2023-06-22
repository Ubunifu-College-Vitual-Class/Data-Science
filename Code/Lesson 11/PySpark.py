import os
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName('Ubunifu_College_PySpark_Lesson').getOrCreate()

# # Reading CSV file
# csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\News Articles Classification.csv"
# df = spark.read.csv(csv_file)

# print(df.count())

# print(df)

# print(df.printSchema())


# # Reading JSON file
# json_file = 'data/json_file.json'
# data = spark.read.json(json_file)



# # Before structuring schema
# csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Stocks_price_final.csv"
# data = spark.read.csv(
#     csv_file,
#     sep = ',',
#     header = True,
#     )
# data.printSchema()

from pyspark.sql.types import *

data_schema = [
               StructField('_c0', IntegerType(), True),
               StructField('symbol', StringType(), True),
               StructField('data', DateType(), True),
               StructField('open', DoubleType(), True),
               StructField('high', DoubleType(), True),
               StructField('low', DoubleType(), True),
               StructField('close', DoubleType(), True),
               StructField('volume', IntegerType(), True),
               StructField('adjusted', DoubleType(), True),
               StructField('market.cap', StringType(), True),
               StructField('sector', StringType(), True),
               StructField('industry', StringType(), True),
               StructField('exchange', StringType(), True),
            ]

final_struc = StructType(fields = data_schema)
csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Stocks_price_final.csv"
data = spark.read.csv(
    csv_file,
    sep = ',',
    header = True,
    schema = final_struc 
    )

# data.printSchema()
# schema = data.schema #This method returns the schema of the data(dataframe).
# print(schema)

# dtypesObject = data.dtypes #It returns n rows as a list.
# print(dtypesObject)

# observations = data.head(3) #It returns n rows as a list.
# print(observations)


# observations = data.show(50) #It displays the first 20 rows by default and it also takes a number as a parameter to display the number of rows of the data
# print(observations)


# firstObservation = data.first() #It returns the first row of the data
# print(firstObservation)


# takeObservation = data.take(10) #It returns the first row of the data
# print(takeObservation)


# describeObject = data.describe() #It computes the summary statistics of the columns with the numeric data type.
# print(describeObject)

# columnsObject = data.columns #It returns a list that contains the column names of the data.
# print(columnsObject)


countObject = data.count() #It returns the count of the number of rows in the data.
print(countObject)


# distinctObject = data.distinct() #It returns the number of distinct rows in the data.
# print(distinctObject)

# printSchemaObject = data.printSchema() # It displays the schema of the data.
# print(printSchemaObject)


# data = data.withColumn('date', data.data)# method takes two parameters column name and data to add a new column to the existing data
# data.show(5)


# data = data.withColumnRenamed('data', 'data_changed')# takes two parameters existing column name and new column name to rename the existing column.
# data.show(5)


# data = data.drop('data')#  takes the column name and returns the data
# data.show(5)


# # Remove Rows with Missing Values
# data.na.drop(subset=["open"])
# data.show()

# # Replacing Missing Values with Mean
# from pyspark.sql import functions as f
# data.na.fill(data.select(f.mean(data['open'])).collect()[0][0])
# data.show()


# # Replacing Missing Values with new values
# old_value = 0
# new_value = 999
# data.na.replace(old_value, new_value)
# data.show()



# ## Selecting Single Column
# data.select('sector').show(5)
# ## Selecting Multiple columns
# data.select(['open', 'close', 'adjusted']).show(5)



# from pyspark.sql.functions import col, lit
# data.filter( (col('data') >= lit('2020-01-01')) & (col('data') <= lit('2020-01-31')) ).show(5)



# ## fetch the data where the adjusted value is between 100.0 and 500.0
# data.filter(data.adjusted.between(100.0, 500.0)).show()


# # It returns 0 or 1 depending on the given condition
# from pyspark.sql import functions as f
# data.select('open', 'close',f.when(data.adjusted >= 200.0, 1).otherwise(0)).show(5)



# # extract the sector names which stars with either M or C using ‘rlike’.
# data.select('sector',data.sector.rlike('^[B,C]').alias('Sector Starting with B or C')).distinct().show()



# # get the average opening, closing, and adjusted stock price concerning industries.
# data.select(['industry','open','close','adjusted']).groupBy('industry').mean().show()


# #display the minimum, maximum, and average; opening, closing, and adjusted stock prices from January 2019 to January 2020 concerning the sectors
# from pyspark.sql.functions import col, lit, min, max, avg
# data.filter( (col('data') >= lit('2019-01-02')) & (col('data') <= lit('2020-01-31')) )\
#     .groupBy("sector") \
#     .agg(min("data").alias("From"), 
#          max("data").alias("To"), 
         
#          min("open").alias("Minimum Opening"),
#          max("open").alias("Maximum Opening"), 
#          avg("open").alias("Average Opening"), 

#          min("close").alias("Minimum Closing"), 
#          max("close").alias("Maximum Closing"), 
#          avg("close").alias("Average Closing"), 

#          min("adjusted").alias("Minimum Adjusted Closing"), 
#          max("adjusted").alias("Maximum Adjusted Closing"), 
#          avg("adjusted").alias("Average Adjusted Closing"), 

#       ).show(truncate=False)


# display a bar graph for the average opening, closing, and adjusted stock price concerning the sector.
import matplotlib.pyplot as plt
# sec_df =  data.select(['sector', 
#                        'open', 
#                        'close', 
#                        'adjusted']
#                      )\
#                      .groupBy('sector')\
#                      .mean()\
#                      .toPandas()

# ind = list(range(12))

# ind.pop(6)

# sec_df.iloc[ind ,:].plot(kind = 'bar', x='sector', y = sec_df.columns.tolist()[1:], 
#                          figsize=(12, 6), ylabel = 'Stock Price', xlabel = 'Sector')
# plt.show()


# industries_x = data.select(['industry', 'open', 'close', 'adjusted']).groupBy('industry').mean().toPandas()
# q  = industries_x[(industries_x.industry != 'Major Chemicals') & (industries_x.industry != 'Building Products')]
# q.plot(kind = 'barh', x='industry', y = q.columns.tolist()[1:], figsize=(10, 50), xlabel='Stock Price', ylabel = 'Industry')
# plt.show()


# from pyspark.sql.functions import col
# tech = data.where(col('sector') == 'Technology').select('data', 'open', 'close', 'adjusted')
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize =(60, 30))
# tech.toPandas().plot(kind = 'line', x = 'data', y='open', xlabel = 'Date Range', ylabel = 'Stock Opening Price',ax = axes[0], color = 'mediumspringgreen')
# tech.toPandas().plot(kind = 'line', x = 'data', y='close',xlabel = 'Date Range', ylabel = 'Stock Closing Price',ax = axes[1], color = 'tomato')
# tech.toPandas().plot(kind = 'line', x = 'data', y='adjusted',xlabel = 'Date Range', ylabel = 'Stock Adjusted Price',ax = axes[2], color = 'orange')
# plt.show()



# # Writing entire data to different file formats
# # CSV
# csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Output\\dataset_pyspark.csv"
# data.write.csv(csv_file)
# # JSON
# json_file = os.path.dirname(os.path.realpath(__file__))+"\\Output\\dataset_pyspark.json"
# data.write.save(json_file, format='json')
# # Parquet
# parquet_file = os.path.dirname(os.path.realpath(__file__))+"\\Output\\dataset_pyspark.parquet"
# data.write.save(parquet_file, format='parquet')

# # Writing selected data to different file formats
# # CSV
# filtered_csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Output\\dataset_pyspark_filtered.csv"
# data.select(['data', 'open', 'close', 'adjusted']).write.csv(filtered_csv_file)
# # JSON
# filtered_json_file = os.path.dirname(os.path.realpath(__file__))+"\\Output\\dataset_pyspark_filtered.json"
# data.select(['data', 'open', 'close', 'adjusted']).write.save(filtered_json_file, format='json')
# # Parquet
# filtered_parquet_file = os.path.dirname(os.path.realpath(__file__))+"\\Output\\dataset_pyspark_filtered.parquet"
# data.select(['data', 'open', 'close', 'adjusted']).write.save(filtered_parquet_file, format='parquet')



































































































































































































