import mysql.connector

mydb = mysql.connector.connect(
  host="mysql-d7e50e4-ypp72448-0713.i.aivencloud.com",
  user="avnadmin",
  password="AVNS_ErEkz-8EvJKAEpX8_ri",
  port = "13302",
  database="mydatabase"
)

mycursor = mydb.cursor()

sql = "INSERT INTO customers (id, name, address) VALUES (%s, %s, %s)"
val = ("2","Priyanshu", "Betul")

mycursor.execute(sql, val)

mydb.commit()

print(mycursor.rowcount, "record inserted.")
