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


def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")

def execute_list_query(connection, sql, val):
    cursor = connection.cursor()
    try:
        cursor.executemany(sql, val)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")



initalconnection = inital_connection("localhost", "root", "1234")
create_database_query = "CREATE DATABASE IF NOT EXISTS python_sql_exercise  "
create_database(initalconnection, create_database_query)

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
create_languages_table = """
 CREATE TABLE IF NOT EXISTS  languages (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(40) NOT NULL
  );
 """

pop_instructors = """
INSERT INTO instructors VALUES
(null,'Mugambi', 'Mundia', 1, NULL, '1985-04-20', 25538907, '+254727310743'),
(null,'Kelvin', 'Otieno', 2, '1', '1995-09-08',  25538906, '+254789123405');
"""

pop_languages = """
INSERT INTO languages VALUES
(null,'Engilsh'),
(null,'French'),
(null,'Germany'),
(null,'Spanish');
"""

execute_query(connection, create_instructors_table) 
execute_query(connection, create_languages_table)
execute_query(connection, pop_instructors)
execute_query(connection, pop_languages)


#Creating Records from Lists 
sql = '''
    INSERT INTO instructors(first_name, last_name, language_1, language_2, dob, national_id, phone_no) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    '''    
val = [
    ( 'Joseph', 'Wambua', 1, 2, '1991-12-23', 23453576, '+254772345678'),
    ( 'Winnie', 'Lee', 1, 2, '1920-12-23', 235323576, '+254772345678'),  
    ( 'Dorcas', 'Mwangi', 2, 2, '1976-02-02', 89785642, '+254443456432')
]
execute_list_query(connection, sql, val)



query_string = """
SELECT 
    i.instructors_id, i.first_name, i.last_name, l.name, i.dob
FROM
    instructors i
        JOIN
    languages l ON l.id = i.language_1;
"""
results = read_query(connection, query_string)
for result in results:
  print(result)

print("---------------------------------------a list of tuples")
from_db = []
for result in results:
  result = result
  from_db.append(result)

print(from_db)
print("---------------------------------------")



print("---------------------------------------a pandas dataframe")
from_db = []
for result in results:
  result = list(result)
  from_db.append(result)

columns = ["Instructor Id", "First name", "Last name", "Language", "Date of birth"]
df = pd.DataFrame(from_db,columns=columns)
print(df)
print("---------------------------------------")


update = """
UPDATE instructors 
SET last_name = 'Eliud' 
WHERE instructors_id = 1;
"""
execute_query(connection, update)

delete_instructors = """
DELETE FROM instructors 
WHERE instructors_id = 5;
"""
execute_query(connection, delete_instructors)


