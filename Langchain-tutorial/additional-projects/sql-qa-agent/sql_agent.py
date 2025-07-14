from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import OpenAI
import os

# Ensure your OPENAI_API_KEY is set as an environment variable

# 1. Set up the database
# In a real application, you would connect to your existing database.
# For this example, we'll create a simple in-memory SQLite database.
db = SQLDatabase.from_uri("sqlite:///sample.db")

# Create some sample tables and data
try:
    db.run("CREATE TABLE employees (id INT, name TEXT, salary INT);")
    db.run("INSERT INTO employees VALUES (1, 'Alice', 100000);")
    db.run("INSERT INTO employees VALUES (2, 'Bob', 80000);")
    db.run("INSERT INTO employees VALUES (3, 'Charlie', 120000);")

    db.run("CREATE TABLE departments (id INT, name TEXT, manager_id INT);")
    db.run("INSERT INTO departments VALUES (1, 'Engineering', 1);")
    db.run("INSERT INTO departments VALUES (2, 'HR', 2);")
except Exception as e:
    print(f"Database already exists or error creating tables: {e}")


# 2. Initialize the LLM
llm = OpenAI(temperature=0)

# 3. Create the SQLDatabaseToolkit
# This toolkit provides the agent with tools for interacting with the SQL database.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 4. Create the SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# 5. Ask a question!
query = "How many employees are there?"
result = agent_executor.run(query)
print(f"Query: {query}\nResult: {result}")

query2 = "Who is the manager of the Engineering department?"
result2 = agent_executor.run(query2)
print(f"\nQuery: {query2}\nResult: {result2}")

# Clean up the database file
os.remove("sample.db")
