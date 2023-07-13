# Objective: Analyze a large Kenyan crime dataset using PySpark to gain insights into crime trends.
# Dataset acquisition
# https://africaopendata.org/datastore/dump/69239d4c-abe9-4213-8035-29ae0607b721?bom=True 

# Load the Dataset: Use PySpark to load the Kenyan crime dataset into a Spark DataFrame.
# Data Exploration: Perform initial exploratory analysis to understand the structure and content of the dataset. Explore the columns, check for missing values, and get a sense of the data distribution.
# Data Cleaning and Transformation: Apply necessary data cleaning and transformation operations to prepare the dataset for analysis. This may include handling missing or incorrect values, converting data types, or filtering out irrelevant data.
# Crime Analysis:
# a. Crime Types: Identify the most common types of crimes reported in Kenya. Use PySpark's aggregation functions to calculate the count of each crime type and display the top crime types.
# b. Crime Trends: Analyze crime trends over time. Group the data by year and calculate the number of reported crimes for each period. Visualize the trends using PySpark's built-in visualization capabilities or by integrating with external plotting libraries.
# c. Crime Analysis: Investigate crime patterns over the year. Identify the year with the highest crime rate and visualize the distribution of crimes along the year period.


# Statistical Analysis: Conduct statistical analysis on the dataset to uncover any significant insights. Calculate descriptive statistics such as mean, median, and standard deviation of crime attributes. 
# Perform hypothesis testing or correlation analysis to explore relationships between variables.
# Data Visualization: Create visualizations using PySpark or external plotting libraries to present the findings from the analysis. Generate charts, graphs, or maps to communicate crime patterns, trends, and other relevant information.
# Performance Optimization: Assuming it is a large dataset , explore PySpark's optimization techniques to improve the performance of your analysis. Utilize techniques like data partitioning, caching, and broadcasting to optimize queries and computations.
# Summary and Insights: Summarize the key findings and insights gained from the analysis of the Kenyan crime dataset.
#  Provide actionable recommendations or suggestions for crime prevention or law enforcement strategies based on the analysis results.

import os
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName('Ubunifu_College_PySpark_Lesson').getOrCreate()

#spark ui can be viewed on http://localhost:4040/jobs

###(A) Load the Dataset: Use PySpark to load the Kenyan crime dataset into a Spark DataFrame.
csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Crime Data.csv"
data = spark.read.csv(
    csv_file,
    sep = ',',
    header = True
    )
print(data.count())



###(B) Data Exploration: Perform initial exploratory analysis to understand the structure and content of the dataset. Explore the columns, check for missing values, and get a sense of the data distribution.

# # structure and schema this method returns the schema of the data(dataframe).
# 1). 
# schema = data.schema 
# print(schema)
# # 2). content
# content = data.show(5) 
# print(content)

# Find Count of Null, None, NaN of All DataFrame Columns
# from pyspark.sql.functions import col,isnan, when, count
# data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()







###(C) Data Cleaning and Transformation:
    
# 1). Remove the entire column
data = data.drop('geo_version')#  takes the column name and returns the data
# data.show(5)




###(D) Crime Analysis:
# 1). Crime Types: Identify the most common types of crimes reported in Kenya.    

# perform the aggregation by crime type and count:
# from pyspark.sql.functions import col, sum
# crime_counts = data.groupBy('crimes').agg(sum('total').alias('count'))
# #sort the results in descending order based on the count:
# sorted_crime_counts = crime_counts.orderBy(col('count').desc())
# #display the top crime types
# sorted_crime_counts.show()




# 2 Crime Trends: Analyze crime trends over time. Group the data by year and calculate the number of reported crimes for each period.
from pyspark.sql.functions import col, sum
# data = data.withColumn('total', col('total').cast('int'))
# crime_counts = data.groupBy('year').agg(sum('total').alias('reported_crimes')).orderBy(col('year').desc())
# crime_counts.show()


# # # Visualize the trend
# # # Convert Spark DataFrame to Pandas DataFrame for plotting
# pandas_df = crime_counts.toPandas()
# # Plotting the trend using a line plot
# import matplotlib.pyplot as plt
# plt.plot(pandas_df['year'], pandas_df['reported_crimes'])
# plt.xlabel('Year')
# plt.ylabel('Reported Crimes')
# plt.title('Trend of Reported Crimes')
# plt.show()





# 3. Crime Analysis: Investigate crime patterns over the year.
# data = data.withColumn('total', col('total').cast('int'))

# crime_counts = data.groupBy('year').agg(sum('total').alias('reported_crimes'))

# # Identify the year with the highest crime rate
# year_with_highest_crime_rate = crime_counts.orderBy(col('reported_crimes').desc()).first()['year']
# print("Year with the highest crime rate:", year_with_highest_crime_rate)
# # Convert Spark DataFrame to Pandas DataFrame for plotting
# pandas_df = crime_counts.toPandas()
# # Plotting the distribution of crimes over the year period
# import matplotlib.pyplot as plt
# plt.bar(pandas_df['year'], pandas_df['reported_crimes'])
# plt.xlabel('Year')
# plt.ylabel('Reported Crimes')
# plt.title('Distribution of Crimes Over the Years')
# plt.show()





### (E) Statistical Analysis: Conduct statistical analysis on the dataset to uncover any significant insights.
# Calculate mean, median, and standard deviation of 'total' column
# from pyspark.sql.functions import col, mean, stddev, expr
# crime_stats = data.agg(mean('total').alias('mean'), expr('percentile_approx(total, 0.5)').alias('median'), stddev('total').alias('stddev'))
# crime_stats.show()




### (F) Perform hypothesis testing or correlation analysis to explore relationships between variables.
# calculating the correlation coefficient between two crime attributes in the given dataset: 'total' (number of crimes) and 'year' (year of occurrence).
# from pyspark.sql.functions import corr
# correlation = data.select(corr('year', 'total').alias('correlation'))
# correlation.show()





### (G) Data Visualization (Repeat if above done)





### (H) Performance Optimization: Assuming it is a large dataset , explore PySpark's optimization techniques to improve the performance of your analysis.
# # Enable dynamic partition pruning
# spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
# # Enable broadcast join threshold
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
# # Load large dataset
# csv_file = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Crime Data.csv"
# df = spark.read.csv(
#     csv_file,
#     sep = ',',
#     header = True
#     )
# # Partition the data
# partitioned_df = df.repartition("crimes")
# # Cache frequently used DataFrame
# partitioned_df.cache()
# # Apply predicate pushdown
# filtered_df = partitioned_df.filter(col("total") == "value")
# # Perform aggregation on the partitioned and filtered data
# aggregated_df = filtered_df.groupBy("crimes").agg(sum("total").alias("total_sum"))
# # Collect the results
# results = aggregated_df.collect()
# # Print the results
# for row in results:
#     print(row["crimes"], row["total_sum"])



# Summary and Insights: Summarize the key findings and insights gained from the analysis of the Kenyan crime dataset.

# Crime Types: The dataset includes various types of crimes reported in Kenya, such as breakings, corruption, criminal damage, dangerous drugs, and economic crimes.
# Crime Distribution: The analysis provides a distribution of reported crimes over the years. The dataset spans multiple years, allowing for observation of trends and patterns in crime rates.
# Year with Highest Crime Rate: By calculating the total reported crimes for each year, it is possible to identify the year with the highest crime rate. The specific year can be determined using statistical measures such as mean, median, or other relevant methods.
# Trend Analysis: Visualizing the trend of reported crimes over the years can reveal patterns or changes in crime rates. Plotting the data can provide insights into whether crime rates are increasing, decreasing, or remaining stable.
# Correlation Analysis: Exploring the relationship between variables, such as the correlation between the year and the number of crimes reported, can provide insights into any potential connections or dependencies between the variables.
# Descriptive Statistics: Calculating descriptive statistics like mean, median, and standard deviation of crime attributes provides a summary of the dataset and helps in understanding the distribution and variation of crime-related data.
# Optimization Techniques: When working with large datasets, employing PySpark's optimization techniques, including data partitioning, caching, predicate pushdown, and parallelism, can significantly improve the performance of analysis and reduce processing time.
# Further Analysis: Beyond the provided analysis, there may be opportunities to explore additional aspects of the dataset, such as crime rates across different regions or crime trends over specific time periods.























































































































































































