
import numpy as np 
import pandas as pd
import os

#Reading CSV file for world food prices
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\AmericasNationalElections.csv")


# Exercise 1

# Using the dataset in Americas National Elections (.CSV attached) , that contains registered voters in the US. 
# It includes demographic information and political preference
# Data Exploration


# How many observations are in the Data Frame?
# How many variables are measured (how many columns)?
# What is the age of the youngest person in the data? The oldest?
# How many days a week does the average respondent watch TV news (round to the nearest tenth)?
# Check for missing values. Are there any?


print(f"The DataFrame contains {len(df)} observations/rows and {len(df.columns)} columns.")
print(f"The youngest respondent is {df.age.min()} years old.")
print(f"The oldest respondent is {df.age.max()} years old.")
print(f"The average respondent watches TV news {df.TVnews.mean():.1f} days a week.")
print( df.isna().sum().sort_values())

# # You could also extract all the requested information using the .describe() method.
# print(df.describe())
# # View data types and if there any missing values.
# df.info()






# Exercise 2
# Data Cleaning & Wrangling
# Rename the educ column education.
# Create a new column called party based on each respondent's answer to PID. party should equal Democrat if the respondent selected either Strong Democrat or Weak Democrat. party will equal Republican if the respondent selected Strong or Weak Republican for PID and Independent if they selected anything else.
# Create a new column called age_group that buckets respondents into the following categories based on their age: 18-24, 25-34, 35-44, 45-54, 55-64, and 65 and over.

# Rename 'educ' column.
df = df.rename(columns={'educ': 'education'})

def getAgeGroup(df):  
  if df['age'] < 25:
    return "18-24"
  elif df['age'] < 35:
    return "25-34"
  elif df['age'] < 45:
    return "35-44"
  elif df['age'] < 55:
    return "45-54"
  elif df['age'] < 65:
    return "55-64"
  else:
    return "65 and over"


def getPartyName(df):
  if df['PID'] < 2:
    return "Democrat"
  elif df['PID'] > 4:
    return "Republican"
  else:
    return "Independent"

# Apply the function to the df and save to a new column 'party.'
df['party'] = df.apply(getPartyName, axis = 1) #he apply() function returns a new DataFrame object after applying the function to its elements
# Apply the function to the df and save to new column 'age_group.'
df['age_group'] = df.apply(getAgeGroup, axis = 1)



# Exercise 3
# Data Filtering
# Use the filtering method to find all the respondents who have the impression that Bill Clinton is moderate or conservative (ClinLR equals 4 or higher). 
#How many respondents are in this subset?
# Among these respondents, how many have a household income less than $50,000 and attended at least some college?

condition = df['ClinLR'] >= 4
respondents = df[condition] #_who_have_the_impression_bill_is_moderate_or_conservative 

print(f"{len(respondents)} respondents have the impression that Bill Clinton is moderate or conservative.")

#income scale
                # 1  - None or less than $2,999
                # 2  - $3,000-$4,999
                # 3  - $5,000-$6,999
                # 4  - $7,000-$8,999
                # 5  - $9,000-$9,999
                # 6  - $10,000-$10,999
                # 7  - $11,000-$11,999
                # 8  - $12,000-$12,999
                # 9  - $13,000-$13,999
                # 10 - $14,000-$14.999
                # 11 - $15,000-$16,999
                # 12 - $17,000-$19,999
                # 13 - $20,000-$21,999
                # 14 - $22,000-$24,999
                # 15 - $25,000-$29,999
                # 16 - $30,000-$34,999
                # 17 - $35,000-$39,999
                # 18 - $40,000-$44,999
                # 19 - $45,000-$49,999
                # 20 - $50,000-$59,999
                # 21 - $60,000-$74,999
                # 22 - $75,000-89,999
                # 23 - $90,000-$104,999
                
#attended some college
                # 1  - Did not attend
                # 2  - Rejected
                # 3  - Finnished
                # 4  - Still schooling

condition_lessin_income = df['income'] < 20

condition_education = df['education'] > 3 

filtered_repondents = df[condition_lessin_income & condition_education] #using Lesson 10,Data Analysis & plotting slide 8
print(f"{len(filtered_repondents)} ({(len(filtered_repondents)/len(df)*100):.2f}%) earn less than $50,000 and attended at least some college.")








































