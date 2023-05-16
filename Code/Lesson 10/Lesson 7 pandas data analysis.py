import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os

#Reading CSV file for Data Science Salaries 2023

#Calculating Basic statistical measurement

df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\ds_salaries.csv") # from Data Science Salaries 2023 link : https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023/download?datasetVersionNumber=1
data_frame = df.head(30)
analysis = df.describe().T

# #Calculating the mean, median, mode, maximum values, minimum values of individual columns 

#Calculating mean salary
# mean = df['salary'].mean()

# #Calculating median salary
# median = df['salary'].median()

# #Calculating mode salary
# mode = df['salary'].mode().iloc[0]

# #Calculating standard deviation
# standard_deviation = df['salary'].std()

# #Calculating minimum values
# minimum = df['salary'].min()

# #Calculating maximum values
# maximum = df['salary'].max()


# print(f" Mean Salary of a Data Science : {mean}")
# print(f" Median Salary a Data Science: {median}")
# print(f" Mode Salary : {mode}")
# print(f" Standard deviation of  Salary : { standard_deviation:.2f}")
# print(f" Maximum of Salary : {maximum}")
# print(f" Minimum of Salary : {minimum}")



# #for display how many umique values for job title column
# sum_unique_job_titles = df['job_title'].nunique()
# print(sum_unique_job_titles)


# #show all unique values
# unique_job_titles = df['job_title'].unique()
# print(unique_job_titles)

# #count of unique values
# count_of_unique_values = df['job_title'].value_counts() 
# print(count_of_unique_values)


# #sort values by job title
# sorted_job_title_values = df.sort_values(by=['job_title'],ascending=True).head(10) 
# print(sorted_job_title_values)



# #display all rows where the job title is ML Engineer
# condition = df['job_title'] == 'ML Engineer'
# filtered_df = df[condition]
# print(filtered_df)


# #Multiple conditions on a dataframe , display all salaries for ML engineers above 100,000
# # first create 3 condition
# condition_job_title = df['job_title'] == 'ML Engineer'

# #change datatype
# df['salary'] = df['salary'].astype('int64') 
# condition_salary = df['salary'] >= 100000

# condition_currency  = df['salary_currency'] == 'USD'

# #pass conditions to the dataframe
# multiple_filtered = df[condition_job_title & condition_salary & condition_currency].head(100)
# print(multiple_filtered)

# #Summarizing or grouping data
# #find maximum values of Job titles and Company location by Salary 
# grouped_df =  df[['job_title','company_location']].groupby(df['salary']).max()
# print(grouped_df)


# #find maximum values of Job titles and Company location by Salary 
# group_agg_df =  df[['job_title','company_location','salary']].groupby(df['job_title']).agg(['count','mean','max'])                     
# print(group_agg_df)


# #grouping by multiple columns
# group_multiple_df =  df[['job_title','company_location','salary','company_size','employee_residence']].groupby(['job_title','company_size','employee_residence']).agg(['count','mean','max'])                     
# print(group_multiple_df)



# #Cross Tabulation (Cross tab)
# cross_tab = pd.crosstab(df['job_title'],df['salary'])
# print(cross_tab)

# #display with percentage than normalize=True parameter 
# cross_tab_normalized = pd.crosstab(df['job_title'],df['salary'] , normalize=True,margins=True,margins_name="Total")
# print(cross_tab_normalized)


#see how the salaries and company location are distributed by Job title 
cross_tab_multiple_columns = pd.crosstab(df['job_title'],[df['company_location'],df['salary']])
print(cross_tab_multiple_columns)


#Data Visualization
# #Line plot
# dict_line = {
#     'year' : [2023,2022,2021,2020],
#     'price':[200,400,100,30]
#     }

# df_line = pd.DataFrame(dict_line)
# #use plot()
# df_line.plot('year','price')

# #Bar plot horizontal
# df['job_title'].head(15).value_counts().plot(kind='bar')

# #Bar plot vertical
# df['job_title'].head(15).value_counts().plot(kind='barh')


# #Pie plot
# df['job_title'].head(15).value_counts().plot(kind='pie')


# #Box plot
# df['salary'].head(3).plot(y=['salary'], kind='box')



# #Box plot - distribution of categorical variables against a numerical variable 
# np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
# fig , ax = plt.subplots(figsize=(6,6))
# df.head(20).boxplot(by = 'job_title', column=['salary'], ax=ax , grid= False)


# #Histogram
# df.head(20).plot(y='salary', kind='hist',bins=10)


# #KDE plot
# df.plot(y='salary',xlim=(0,200000), kind='kde')


# #Scatter plot
# df.head(10).plot(x='job_title',y='salary',kind='scatter')



































































































































































