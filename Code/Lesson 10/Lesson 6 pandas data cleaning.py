
import numpy as np 
import pandas as pd
import os

#Reading CSV file for world food prices

df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Wfp_food_prices_ken.csv")
data_frame = df.head(30)
print(data_frame)


#Drop unwanted columns
# df.drop(['admin2'], axis=1 , inplace=True)
# print(df.info())


#Change Columns name
#create new df_col dataframe from df.copy() method.
df_col = df.copy()
#rename columns name
df_col.rename(columns={"date" : "transaction date" , "admin1" : "constituency" , "admin2": "county","price": "buying price"} , inplace = True)
# renamed_df = df_col.head(3)
df = df_col




# Adding a new column to a Data Frame
df['selling price'] = 10
df_with_new_column = df.head(3) #nb -> to check if additon of a column in between
print(df_with_new_column)



#String value change or replace
df.loc[df.constituency == "Coast", 'constituency'] = 'The Coast' 
print(df)


#Datatype change
# # change object type to datefime64 format

df = df.drop(df.index[[0,0]]) # drop the first row as it contains strings * bad data for date*

#print(df.info())

# df['transaction date'] = df['transaction date'].astype('datetime64[ns]')
# # change float64 to float32 of Referal columns
# df['buying price'] = df['buying price'].astype('float32')
# print(df.info())



#Remove duplicate
# # Display duplicated entries 
duplicates = df.duplicated().sum()


# # duplicate rows dispaly, keep arguments will--- 'first', 'last' and False
duplicate_value = df.duplicated(keep='first')
new_filtered_df = df.loc[duplicate_value, :]
print(new_filtered_df)

# # dropping ALL duplicate values
df.drop_duplicates(keep = 'first', inplace = True)
print(df);


# Handling missing values

#Display missing values information
# missing_values = df.isna().sum().sort_values(ascending=True)
# print(missing_values);


# Display missing values information

# df copy to df_copy
df_new = df.copy()


#Delete Nan rows of "constituency","county","market" Columns
# df_new.dropna(subset = ["constituency","county","market"], inplace=True)
# df = df_new.copy() #copy back to df for presentation

# print(df)

# Delete entire columns
# df_new.drop(columns=['buying price'], inplace=True)
# df_new.isna().sum().sort_values(ascending=False)

# df_dropped_column = df_new.copy()
# print(df_dropped_column)



# Impute missing values
# Method 1 —  Impute fixed values like 0, ‘Unknown’ or ‘Missing’ etc
# df['county'].fillna('Unknown', inplace=True)
# df_imputed_replaced = df.copy()
# print(df_imputed_replaced)


# Method 2 — Impute Mean, Median, and Mode

# Impute Mean in buying price columns
# df = df.drop(df.index[[0,0]]) # drop the first row as it contains strings * bad data for date*
# df['buying price'] = df['buying price'].astype('float32') # convert to appropriate datatype
# mean_price = df['buying price'].mean()
# df['buying price'].fillna(mean_price, inplace=True)
# df_imputed_mean_price_column = df.copy()

#Imputing using using median
# median_price = df['buying price'].median()

# df['buying price'].fillna(median_price, inplace=True)
# df_imputed_median_price_column = df.copy()
# print(df_imputed_median_price_column)

#Imputing using using mode
# mode_price = df['buying price'].mode().iloc[0]
# df['buying price'].fillna(mode_price, inplace=True)
# df_imputed_mode_price_column = df.copy()
# print(df_imputed_mode_price_column)



#Method 3 — Imputing forward fill or backfill by ffill and bfill. 
df['buying price'].fillna(method='ffill', inplace=True)
df_imputed_ffill = df.copy()
print(df_imputed_ffill)


# Memory Management


#Change datatype e.g using df['buying price'] = df['buying price'].astype('float32')  
df_memory = df.copy()
memory_usage = df_memory.memory_usage(deep=True)
memory_usage_in_mbs = round(np.sum(memory_usage / 1024 ** 2), 3)
print(f" Total memory taking df_memory dataframe is : {memory_usage_in_mbs:.2f} MB ")


#Change object to category data types
# df_memory[ df_memory.select_dtypes(['object']).columns] = df_memory.select_dtypes(['object']).apply(lambda x: x.astype('category'))
# df_memory.info(memory_usage="deep")


#Change int64 or float64 to int 32, 16
# Change Referal column datatypes
# df_memory['latitude'] = df_memory['latitude'].astype('float16')
# df_memory['buying price'] = df_memory['buying price'].astype('float16')
# df_memory.info(memory_usage="deep")

































































































