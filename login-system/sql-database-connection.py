import mysql.connector

# Step 1: Connect to the MySQL server
mydb = mysql.connector.connect(
  host="mysql-486c28e-yogieee6094-c2ae.j.aivencloud.com",
  user="avnadmin",
  password="AVNS_6KdboIlyiUKBqA0d0O7",
  port=23230
)

mycursor = mydb.cursor()

# Step 2: Create the database if it doesn't exist
mycursor.execute("CREATE DATABASE IF NOT EXISTS loginsystem")

# Step 3: Connect to the 'loginsystem' database
mydb.database = "loginsystem"

# Step 4: Create the 'user' table if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

mycursor.execute(create_table_query)

print("Database and table setup complete.")
