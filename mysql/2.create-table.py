import mysql.connector

mydb = mysql.connector.connect(
  host="mysql-d7e50e4-ypp72448-0713.i.aivencloud.com",
  user="avnadmin",
  password="AVNS_ErEkz-8EvJKAEpX8_ri",
  port = "13302",
  database="mydatabase"
)

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE customers (id INT(20) PRIMARY KEY ,name VARCHAR(255), address VARCHAR(255))")
