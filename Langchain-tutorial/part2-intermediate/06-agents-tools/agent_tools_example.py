import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import OpenAI
from langchain.tools import Tool

# Ensure your OPENAI_API_KEY and SERPAPI_API_KEY are set as environment variables

# 1. Initialize the LLM
llm = OpenAI(temperature=0)

# 2. Define the Tools
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for searching the web and getting concise answers to factual questions.",
    )
]

# 3. Create the Agent
# We get a prompt template from the hub that is designed for ReAct agents.
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

# 4. Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 5. Run the Agent
query = "What is the current weather in San Francisco?"
print(f"Running query: {query}")
response = agent_executor.invoke({"input": query})
print(f"Agent's response: {response}")

# Another example
query2 = "What are the main differences between Langchain and llamaindex?"
print(f"\nRunning query: {query2}")
response2 = agent_executor.invoke({"input": query2})
print(f"Agent's response: {response2}")
