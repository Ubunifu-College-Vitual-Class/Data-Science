# Using Nation Africa, Extract images and save them in your mysql table as a blob column for later use in image recognition (Use reference lesson 5 python-sql for mysql inserts)
# Link : https://nation.africa/kenya

import mysql.connector
import requests
from bs4 import BeautifulSoup
from mysql.connector import Error

from urllib.request import urlopen # use request for the python urllib library
from bs4 import BeautifulSoup

#prepare a connection to mysql
def inital_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL server connection successful")
        cursor = connection.cursor()  
        cursor.execute("SELECT VERSION()")
        result = cursor.fetchall()
        print("MySQL version: %s " %  result[0]) 
    except Error as err:
        print(f"Error: '{err}'")

    return connection

def create_server_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")

    except Error as err:
        print(f"Error: '{err}'")
    return connection

#function to create database
def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")



#initialize the connection
initalconnection = inital_connection("localhost", "root", "1234")

#create the database
create_database_query = "CREATE DATABASE IF NOT EXISTS python_sql_exercise "
create_database(initalconnection, create_database_query)

#connects to the newly created database
connection = create_server_connection("localhost", "root", "1234","python_sql_exercise")

#create a table nation_africa_images to store the blob data
create_instructors_table = """
CREATE TABLE IF NOT EXISTS nation_africa_images (
  id INT PRIMARY KEY AUTO_INCREMENT,
  image BLOB
  );
 """

cursor = connection.cursor()
cursor.execute(create_instructors_table)
connection.commit()
print("Query successful")









#do the webscrapping

base_url = "https://nation.africa/kenya"
page = requests.get(base_url) # grab the page

soup = BeautifulSoup(page.content, "html.parser") # create your beautiful soup object

imagescrapped = soup.select('img')# scrap all images

for image in imagescrapped:
 
    if image.has_key('src'): #for images set with source attribute
        images_url = image['src']       
    elif image.has_key('data-src'): #images set with data attribute
        images_url = image['data-src'] 

    fulimageurl = "https://nation.africa"+images_url
    #print("https://nation.africa"+images_url) 
    #inserting blob data

    #data=requests.get(fulimageurl)   # read image
    #photo=data.content

    sql = """
    INSERT INTO nation_africa_images VALUES
    (null,'"""+fulimageurl+"""');
    """ 
    execute_query(connection, sql)




