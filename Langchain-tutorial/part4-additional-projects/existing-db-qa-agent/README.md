# Project: Q&A Agent for an Existing SQL Database

This project demonstrates a more realistic scenario where you build a LangChain agent that connects to a pre-existing SQL database to answer questions in natural language.

## Objective

The goal is to create an agent that can:
1.  Connect to an existing SQL database (e.g., PostgreSQL, MySQL, SQL Server).
2.  Inspect the database's schema to understand the available tables and their columns.
3.  Take a natural language question from the user (e.g., "What are the top 5 best-selling products?").
4.  Convert the question into a valid SQL query for the given database schema.
5.  Execute the query and return a natural language answer.

## Step-by-Step Implementation

### 1. Prerequisites

*   **An existing SQL database:** You need to have a database running with some tables and data in it.
*   **Database driver:** You need to install the appropriate Python driver for your database.
    *   For PostgreSQL: `pip install psycopg2-binary`
    *   For MySQL: `pip install mysql-connector-python`
    *   For SQL Server: `pip install pyodbc`

### 2. Configure Database Connection

You will need to create a database URI that tells LangChain how to connect to your database. The format is typically:
`dialect+driver://username:password@host:port/database`

**Examples:**
-   **PostgreSQL:** `postgresql+psycopg2://user:password@localhost:5432/mydatabase`
-   **MySQL:** `mysql+mysqlconnector://user:password@localhost:3306/mydatabase`
-   **SQL Server:** `mssql+pyodbc://user:password@myserver/mydatabase?driver=ODBC+Driver+17+for+SQL+Server`

It is highly recommended to store these connection details securely (e.g., using environment variables) rather than hard-coding them.

### 3. Create the Main Script

Now, let's create the main Python script for our application.

**`existing_db_agent.py`**

```python
import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import OpenAI

# --- IMPORTANT ---
# 1. Ensure you have installed the necessary database driver (e.g., pip install psycopg2-binary)
# 2. Set your database connection URI and OpenAI API key as environment variables.

# 1. Get database URI from environment variable
# Example for PostgreSQL: "postgresql+psycopg2://user:password@localhost:5432/mydatabase"
db_uri = os.environ.get("DATABASE_URI")
if not db_uri:
    raise ValueError("DATABASE_URI environment variable not set.")

# 2. Set up the database connection
db = SQLDatabase.from_uri(db_uri)

# 3. Initialize the LLM
llm = OpenAI(temperature=0)

# 4. Create the SQLDatabaseToolkit
# This toolkit provides the agent with tools to interact with your database.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 5. Create the SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# 6. Ask a question about your data!
# Replace this with a question relevant to your database schema.
query = "Describe the 'products' table for me."
result = agent_executor.run(query)
print(f"Query: {query}\nResult: {result}")

query2 = "How many users are registered?"
result2 = agent_executor.run(query2)
print(f"\nQuery: {query2}\nResult: {result2}")
```

### 4. How to Run the Project

1.  Make sure you have your `OPENAI_API_KEY` and `DATABASE_URI` environment variables set.
2.  Navigate to this directory in your terminal.
3.  Run the script:

    ```bash
    python existing_db_agent.py
    ```

4.  The agent will inspect your database schema, generate and execute SQL queries based on your questions, and provide answers in natural language.

This project is a powerful demonstration of how to create a natural language interface for any existing SQL database, making data more accessible to non-technical users.
