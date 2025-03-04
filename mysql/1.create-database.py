import mysql.connector

mydb = mysql.connector.connect(
  host="mysql-d7e50e4-ypp72448-0713.i.aivencloud.com",
  user="avnadmin",
  password="AVNS_ErEkz-8EvJKAEpX8_ri",
  port = "13302"
)

mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE mydatabase")

