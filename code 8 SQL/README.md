# 初始化
1. Install MySQL Connector Python using pip. Open Anacoda Prompt and then use the following pip command to install MySQL Connector python.
```
pip install mysql-connector-python
```

2. Use the mysql.connector.connect() method of MySQL Connector Python with required parameters to connect MySQL.

3. Use the connection object returned by a connect() method to create a cursor object to perform Database Operations.

4. The cursor.execute() to execute SQL queries from Python.

5. Close the Cursor object using a cursor.close() and MySQL database connection using connection.close() after your work completes.

6. Catch Exception if any that may occur during this process.

一个简单的链接MySQL例子：
```
import mysql.connector
from mysql.connector import Error
try:
	connection = mysql.connector.connect(host='localhost’, database='Computer’, 	user=‘pyDurham’, password=‘iolx1zi@#2')
	if connection.is_connected():
		db_Info = connection.get_server_info()
		print("Connected to MySQL Server version ", db_Info)
		cursor = connection.cursor()
		cursor.execute("select database();")
		record = cursor.fetchone()
		print("You're connected to database: ", record)
except Error as e:
	print("Error while connecting to MySQL", e)
finally:
	if (connection.is_connected()):
	cursor.close()
	connection.close()
	print("MySQL connection is closed")
```
你需要用cursor创建一个cursor对象，这个对象才有.execute来执行SQL queries。cursor是游标，它能帮助你在得到第一次结果之后，进一步得到第二次的结果。用游标的execute来执行SQL，得到的结果会储存起来，然后你就可以进行第二次操作。fetchone()一次获取一条记录（元组格式），每获取一条游标往前移动一格；fetchall()获取所有数据，返回元组列表。
# 创建一个table
（Python不会检查MySQL语法，所以相关语法你必须自己检查好）
```
import mysql.connector
from mysql.connector import Error
try:
	connection = mysql.connector.connect(host='localhost’, database=‘Computer’, user=‘pyDurham’, password=‘iolx1zi@#2’)
	mySql_Create_Table_Query = """CREATE TABLE Laptop ( id int(11) NOT NULL, name varchar(250) NOT NULL,
price float NOT NULL, purchase_date Date NOT NULL, PRIMARY KEY (id)) """
	cursor = connection.cursor()
	result = cursor.execute(mySql_Create_Table_Query)
	print("Laptop Table created successfully ")
except mysql.connector.Error as error:
	print("Failed to create table in MySQL: {}".format(error))
finally:
	if (connection.is_connected()):
		cursor.close()
		connection.close()
		print("MySQL connection is closed")
```
# insert一个记录
1) First, Establish a MySQL database connection in Python.

2) Then, Define the SQL INSERT Query (here you need to know the table’s column details).

3) Execute the INSERT query using the cursor.execute()and get a number of rows affected.

4) After the successful execution of a query, Don’t forget to commit your changes to the database.

5) Close the MySQL database connection.

6) Most important, Catch SQL exceptions if any.

7) At last, verify the result by selecting data from the MySQL table.

```
import mysql.connector
from mysql.connector import Error
try:
	connection = mysql.connector.connect(host='localhost’, database=‘Computer’, user=‘pyDurham’, password=‘iolx1zi@#2’)
	mySql_insert_query = """INSERT INTO Laptop (id, name, price, purchase_date) VALUES
(10, 'Lenovo ThinkPad P71', 6459, '2019-08-14') """
	cursor = connection.cursor()
	cursor.execute(mySql_insert_query)
	connection.commit()
	print(cursor.rowcount, "Record inserted successfully into Laptop table")
except mysql.connector.Error as error:
	print("Failed to insert record into Laptop table {}".format(error))
finally:
	if (connection.is_connected()):
		cursor.close()
		connection.close()
		print("MySQL connection is closed")
```

```
import mysql.connector
from mysql.connector import Error
def insertVariblesIntoTable(id, name, price, purchase_date):
	try:
		connection = mysql.connector.connect(host='localhost’, database=‘Computer’, user=‘pyDurham’, password=‘iolx1zi@#2’)
		cursor = connection.cursor()
		mySql_insert_query = "INSERT INTO Laptop (id, name, price, purchase_date) VALUES (%s, %s, %s, %s) "
		recordTuple = (id, name, price, purchase_date)
		cursor.execute(mySql_insert_query, recordTuple)
		connection.commit()
		print("Record inserted successfully into Laptop table")
	except mysql.connector.Error as error:
		print("Failed to insert into MySQL table {}".format(error))
	finally:
		if (connection.is_connected()):
			cursor.close()
			connection.close()
			print("MySQL connection is closed")
insertVariblesIntoTable(2, 'Area 51M', 6999, '2019-04-14')
insertVariblesIntoTable(3, 'MacBook Pro', 2499, '2019-06-20')
```

# SQLAlchemy
SQLAlchemy is the Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL. SQLAlchemy provides a full suite of well known enterprise-level persistence patterns, designed for efficient and high-performing database access, adapted into a simple and Pythonic domain language.

![](1.png)

The string form of the URL is dialect[+driver]://user:password@host/dbname[?key=value..], where dialect is a DBMS name such as mysql, oracle, postgresql, etc., and driver the name of a DBAPI, such as psycopg2, pyodbc, cx_oracle, etc. Alternatively, the URL can be an instance of URL.

```
from sqlalchemy import create_engine
sqlEngine = create_engine("mysql+pymysql://usr:password@myeusql.dur.ac.uk/user_world")
dbConnection = sqlEngine.connect()
```
# pandas.read_sql
创建把MySQL数据库转换成pandas，你需要：
Obtain an SQLAlchemy engine object to connect to the MySQL database server by providing required credentials.

Using the engine object, connect to the MySQL server by calling the connect() method.

Create a dictionary of python lists. The python lists correspond to column data and the keys of the python dict correspond to column names.

Create a dataframe by calling the pandas dataframe constructor and passing the python dict object as data.

Invoke to_sql() method on the pandas dataframe instance and specify the table name and database connection. This creates a table in MySQL database server and populates it with the data from the pandas dataframe.

```
from sqlalchemy import create_engine
import pymysql
import pandas as pd
userVitals = {"UserId":["xxxxx", "yyyyy", "zzzzz", "aaaaa", "bbbbb", "ccccc", "ddddd"],"UserFavourite":["Greek Salad", "Philly Cheese Steak", "Turkey Burger","Crispy Orange Chicken", "Atlantic Salmon", "Pot roast", "Banana split"],"MonthlyOrderFrequency":[5, 1, 2, 2, 7, 6, 1],"HighestOrderAmount":[30, 20, 16, 23, 20, 26, 9],"LastOrderAmount":[21,20,4,11,7,7,7],"LastOrderRating":[3,3,3,2,3,2,4],"AverageOrderRating":[3,4,2,1,3,4,3],"OrderMode":["Web", "App", "App", "App", "Web", "Web", "App"],"InMedicalCare":["No", "No", "No", "No", "Yes", "No", "No"]};
tableName = "UserVitals"
dataFrame = pd.DataFrame(data=userVitals)
sqlEngine =create_engine('mysql+pymysql://user:password@myeusql.dur.ac.uk/db_name)
dbConnection = sqlEngine.connect()
try:
	frame = dataFrame.to_sql(tableName, dbConnection, if_exists='fail');
except ValueError as vx:
	print(vx)
except Exception as ex:
	print(ex)
else:
	print("Table %s created successfully."%tableName);
finally:
	dbConnection.close()
```
如果只是读取就简单多了，你需要：

Call read_sql() method of the pandas module by providing the SQL Query and the SQL Connection object to get data from the MySQL database table.

The database connection to MySQL database server is created using sqlalchemy.

read_sql() method returns a pandas dataframe object. The frame will have the default-naming scheme where the rows start from zero and get incremented for each row. The columns will be named after the column names of the MySQL database table.

```
from sqlalchemy import create_engine
import pymysql
import pandas as pd
sqlEngine =create_engine('mysql+pymysql://user:password@myeusql.dur.ac.uk/db_name)
dbConnection = sqlEngine.connect()
Frame = pd.read_sql("select * from test.uservitals", dbConnection);
pd.set_option('display.expand_frame_repr', False)
print(frame)
dbConnection.close()
```
pandas.read_sql_table这个函数是Read SQL database table into a DataFrame。而pandas.read_sql_query 的作用是Read SQL query into a DataFrame。而pandas.read_sql可以Read SQL query or database table into a DataFrame.

# MySQLdb
