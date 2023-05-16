
import pandas as pd
import os

#Reading CSV file

# df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Wfp_food_prices_ken.csv")
# data_frame = df.head(3)
# print(data_frame)

#Reading from URL
# url="https://data.humdata.org/dataset/e0d3fba6-f9a2-45d7-b949-140c455197ff/resource/517ee1bf-2437-4f8c-aa1b-cb9925b9d437/download/wfp_food_prices_ken.csv"
# df_url = pd.read_csv(url)
# data_frame = df_url.head(3)
# print(data_frame)


#Write CSV file
# df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Wfp_food_prices_ken.csv")#read the file to create a dataframe
# data_frame = df.head(3)
# data_frame.to_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\to_csv\\csv_from_url.csv")
# data_frame.to_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\to_csv\\write_to_text.txt")


#Read text file (download from kaggle : link : https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs : Huge Stock Market Dataset  )
# df_txt = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Huge_Stock_Market_Dataset.txt", sep=",")
# data_frame = df_txt.head(10)
# print(data_frame)

#Read Excel file (Standards of Performance for Greenhouse Gas Emissions from Existing Sources link: https://www.datarefuge.org/dataset/standards-of-performance-for-greenhouse-gas-emissions-from-existing-sources )
df_excel = pd.read_excel(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Epa-hq-oar-2013-0602-0465.xlsx", sheet_name='All Constraints')
data_frame = df_excel.head(50)
print(data_frame)
#write df to excel
# data_frame.to_excel(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\to_csv\\Greenhouse.xlsx",sheet_name='Prepared data')

#Retrieving rows from a data frame.
# data_frame = df_excel.tail(6) 
# print(data_frame)

#Display random 7 sample rows
# data_frame = df_excel.sample(7) 
# print(data_frame)


#Retrieving information about the data frame
# info = df_excel.info() 
# print(info)

#Display data types of each column 
# info = df_excel.dtypes
# print(info)


#Display data types values counting.
# info = df_excel.dtypes.value_counts()
# print(info)



#Display data types values counting.
# print(df_excel.shape)

#Display columns name and data
# columns =  df_excel.columns
# print(columns)


#Display contraint name column for first 3 rows
# contraine_name_column =  df_excel['Constraint Name'].head(3)
# print(contraine_name_column)



#display first 4 rows of Constraint name , Constraint # , Constraint ID , Constraint Type 
# top_four_rows =  df_excel[['Constraint Name', 'Constraint #', 'Constraint ID', 'Constraint Type']].head(4)
# print(top_four_rows)


#Retrieving a Range of Rows
# range_of_rows =  df_excel[2:7]
# print(range_of_rows)


#displaying the last two rows
# range_of_rows =  df_excel[-2:]
# print(range_of_rows)


#access more than one row, use double brackets and specify the indexes, separated by commas
# extracting_rows_illoc =  df_excel.iloc[[0,2,5,10]]
# print(extracting_rows_illoc)


#Specify columns by including their indexes in another list
# extracting_rows_illoc =  df_excel.iloc[[0, 2,10], [0, 1]]
# print(extracting_rows_illoc)


#specify a slice of the DataFrame with from and to indexes, separated by a colon
# extracting_rows_illoc =  df_excel.iloc[0 : 2]
# print(extracting_rows_illoc)























































































