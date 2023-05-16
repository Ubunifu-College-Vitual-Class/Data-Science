import mysql.connector
from mysql.connector import Error
import pandas as pd

def inital_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL Database connection successful")
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

#initialize the connection
initalconnection = inital_connection("localhost", "root", "1234")

#create the database
create_database_query = "CREATE DATABASE IF NOT EXISTS python_sql_exercise  "
create_database(initalconnection, create_database_query)




#connects to the newly created database
connection = create_server_connection("localhost", "root", "1234","python_sql_exercise")

create_instructors_table = """
CREATE TABLE IF NOT EXISTS instructors (
  instructors_id INT PRIMARY KEY AUTO_INCREMENT,
  first_name VARCHAR(40) NOT NULL,
  last_name VARCHAR(40) NOT NULL,
  language_1  INT(3) NOT NULL,
  language_2 INT(3),
  dob DATE,
  national_id INT UNIQUE,
  phone_no VARCHAR(20)
  );
 """

cursor = connection.cursor()
cursor.execute(create_instructors_table)
connection.commit()
print("Query successful")

#inserting records
pop_instructors = """
INSERT INTO instructors VALUES
(null,'Mugambi', 'Mundia', 1, NULL, '1985-04-20', 25538907, '+254727310743'),
(null,'Kelvin', 'Otieno', 2, '1', '1995-09-08',  25538906, '+254789123405');
"""

#cursor = connection.cursor()
#cursor.execute(pop_instructors)
#connection.commit()
#print("Query successful")

#fetch data from the table instructor


query_string = """
SELECT * FROM  instructors
"""
cursor = connection.cursor()
cursor.execute(query_string)
results = cursor.fetchall()

#for result in results:
 #print(result)

    #from_db = []
   # for result in results:
        #result = result
    #from_db.append(result)

    #print(from_db)
 #  

from_db = []
for result in results:
  result = list(result)
  from_db.append(result)

columns = ["instructors_id", "first_name", "last_name", "language_1", "language_2","dob","national_id","phone_no"]
df = pd.DataFrame(from_db,columns=columns)
print(df)


 