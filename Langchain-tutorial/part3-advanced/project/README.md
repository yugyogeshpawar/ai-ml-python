# Part 3 Project: Deploying a Research Agent with LangServe

In this final project, we will combine everything we've learned to build a sophisticated research agent and deploy it as a production-ready, streaming API using LangServe.

## Objective

The goal is to create and deploy an agent that can:
1.  Use a web search tool to find information on a given topic.
2.  Summarize the findings into a concise report.
3.  Be deployed as a REST API using LangServe.
4.  Stream the final summary to the client in real-time.

## Step-by-Step Implementation

### 1. Define the Agent and Tools

First, we'll create the agent and the tools it needs. For this project, a simple web search tool is sufficient.

### 2. Create the LangServe Application

Next, we'll create a FastAPI application and use LangServe to deploy our agent.

**`research_agent_api.py`**

```python
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
```

### 3. How to Run the Project

1.  Make sure you have your `OPENAI_API_KEY` and `SERPAPI_API_KEY` environment variables set.
2.  Install the required packages:

    ```bash
    pip install langchain langchain-openai langchainhub fastapi uvicorn serpapi
    ```
3.  Navigate to the `project` directory in your terminal.
4.  Run the server:

    ```bash
    uvicorn research_agent_api:app --reload
    ```

### 4. Interact with the Deployed Agent

*   **Interactive Playground:** Go to `http://127.0.0.1:8000/docs`. You can test the `/research-agent/invoke` and `/research-agent/stream` endpoints here.
*   **Streaming with `curl`:** To see the real-time streaming in action, use the `/stream` endpoint from your command line:

    ```bash
    curl -X POST http://127.0.0.1:8000/research-agent/stream \
    -H "Content-Type: application/json" \
    -d '{"input": "Summarize the latest news about the impact of AI on the job market."}'
    ```
    You will see the agent's final summary streamed to your terminal as it's being generated.

This project is a culmination of all the concepts you've learned in this tutorial. You've built a powerful, data-aware agent and deployed it as a production-ready, streaming API.

## Congratulations!

You have now completed the LangChain tutorial, from the fundamentals to advanced deployment. You have the skills and knowledge to build a wide range of sophisticated applications with LangChain. The possibilities are endless, and the journey is just beginning. Happy building!
