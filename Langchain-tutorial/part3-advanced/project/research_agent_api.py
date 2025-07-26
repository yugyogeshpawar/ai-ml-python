import os
from fastapi import FastAPI
from langserve import add_routes
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import uvicorn

# Ensure your OPENAI_API_KEY and SERPAPI_API_KEY are set

# 1. Define the Tools
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for searching the web and getting concise answers to factual questions.",
    )
]

# 2. Initialize the LLM and Agent
llm = ChatOpenAI(temperature=0, model="gpt-4")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 3. Create the FastAPI app
app = FastAPI(
    title="Research Agent API",
    version="1.0",
    description="An API for a research agent that can browse the web and summarize findings.",
)

# 4. Add the agent to the app using LangServe
add_routes(
    app,
    agent_executor,
    path="/research-agent",
)

# 5. Add a main block to run the server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
