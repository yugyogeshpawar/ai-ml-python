import os
from fastapi import FastAPI
from langserve import add_routes
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchainhub import serper
import uvicorn

# Ensure your OPENAI_API_KEY and SERPAPI_API_KEY are set

# 1. Define the Tools
tools = [
    Tool(
        name="Web Search",
        func=serper.run,
        description="Useful for searching the web and getting concise answers to factual questions.",
    )
]

# 2. Initialize the LLM and Agent
llm = ChatOpenAI(temperature=0, model="gpt-4")
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
)

# 3. Create the FastAPI app
app = FastAPI(
    title="Research Agent API",
    version="1.0",
    description="An API for a research agent that can browse the web and summarize findings.",
)

# 4. Add the agent to the app using LangServe
add_routes(
    app,
    agent,
    path="/research-agent",
)

# 5. Add a main block to run the server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
