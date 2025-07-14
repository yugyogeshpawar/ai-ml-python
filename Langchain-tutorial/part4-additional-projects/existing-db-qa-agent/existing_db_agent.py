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
    raise ValueError("DATABASE_URI environment variable not set. Please set it before running the script.")

# 2. Set up the database connection
try:
    db = SQLDatabase.from_uri(db_uri)
except Exception as e:
    print(f"Error connecting to the database: {e}")
    print("Please ensure your DATABASE_URI is correct and the database is running.")
    exit()

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
print("--- Example 1: Describing a table ---")
query = "Describe the 'products' table for me."
try:
    result = agent_executor.run(query)
    print(f"Query: {query}\nResult: {result}")
except Exception as e:
    print(f"An error occurred: {e}")


print("\n--- Example 2: Counting rows ---")
query2 = "How many users are registered?"
try:
    result2 = agent_executor.run(query2)
    print(f"Query: {query2}\nResult: {result2}")
except Exception as e:
    print(f"An error occurred: {e}")
