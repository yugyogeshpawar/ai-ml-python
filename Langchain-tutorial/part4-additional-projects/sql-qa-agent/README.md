# Project: SQL Database Q&A Agent

This project demonstrates how to build a LangChain agent that can connect to a SQL database and answer natural language questions about the data.

## Objective

The goal is to create an agent that can:
1.  Connect to a sample SQL database (e.g., SQLite).
2.  Inspect the database schema to understand the tables and columns.
3.  Take a natural language question from the user (e.g., "How many employees are there?").
4.  Convert the natural language question into a SQL query.
5.  Execute the SQL query against the database.
6.  Interpret the result of the query and provide a natural language answer to the user.

## Step-by-Step Implementation

### 1. Install Necessary Libraries

We'll need `langchain-experimental` for the SQL agent components.

```bash
pip install langchain-experimental
```

### 2. Create a Sample Database

For this project, we'll create a simple SQLite database with a few tables.

### 3. Create the Main Script

Now, let's create the main Python script for our application.

**`sql_agent.py`**

```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import OpenAI

# 1. Set up the database
# In a real application, you would connect to your existing database.
# For this example, we'll create a simple in-memory SQLite database.
db = SQLDatabase.from_uri("sqlite:///sample.db")

# Create some sample tables and data
db.run("CREATE TABLE employees (id INT, name TEXT, salary INT);")
db.run("INSERT INTO employees VALUES (1, 'Alice', 100000);")
db.run("INSERT INTO employees VALUES (2, 'Bob', 80000);")
db.run("INSERT INTO employees VALUES (3, 'Charlie', 120000);")

db.run("CREATE TABLE departments (id INT, name TEXT, manager_id INT);")
db.run("INSERT INTO departments VALUES (1, 'Engineering', 1);")
db.run("INSERT INTO departments VALUES (2, 'HR', 2);")


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
```

### 4. How to Run the Project

1.  Make sure you have your `OPENAI_API_KEY` environment variable set.
2.  Navigate to this directory in your terminal.
3.  Run the script:

    ```bash
    python sql_agent.py
    ```

4.  The script will print the agent's thought process as it converts your questions into SQL queries, executes them, and generates the final answers.

This project demonstrates a powerful real-world use case for LangChain agents: creating natural language interfaces for structured data.
