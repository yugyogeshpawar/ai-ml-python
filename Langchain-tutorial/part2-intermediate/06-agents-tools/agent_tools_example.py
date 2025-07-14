import os
from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAI
from langchainhub import serper

# Ensure your OPENAI_API_KEY and SERPAPI_API_KEY are set as environment variables

# 1. Initialize the LLM
llm = OpenAI(temperature=0)

# 2. Define the Tools
# In this example, we're using a web search tool from SerpAPI.
# You can explore other tools in the LangChain documentation.
tools = [
    Tool(
        name="Web Search",
        func=serper.run,
        description="Useful for searching the web and getting concise answers to factual questions.",
    )
]

# 3. Initialize the Agent
# We're using the "zero-shot-react-description" agent type, which relies on tool descriptions.
# The agent uses the LLM to decide which tool to use based on the query and the tool descriptions.
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,  # Set verbose to True to see the agent's reasoning steps
)

# 4. Run the Agent
# Provide a query to the agent, and it will use the available tools to try to answer it.
query = "What is the current weather in San Francisco?"
print(f"Running query: {query}")
response = agent.run(query)
print(f"Agent's response: {response}")

# Another example
query2 = "What are the main differences between Langchain and llamaindex?"
print(f"\nRunning query: {query2}")
response2 = agent.run(query2)
print(f"Agent's response: {response2}")
