# 6. Agents and Tools: Unleashing the Power of Autonomous Decision-Making

We've built chains that can load data, split it, retrieve relevant chunks, and even remember past conversations. Now, we're ready for the final and most advanced concept in this section: **Agents and Tools**.

This is where LangChain truly shines, allowing you to build applications that can:
*   **Reason about tasks:** Understand what needs to be done to achieve a goal.
*   **Choose the right tools:** Select the appropriate tools from a set of available options.
*   **Execute actions:** Use those tools to interact with the outside world.
*   **Observe results:** Analyze the output of the tools.
*   **Repeat:** Iterate through this process until the goal is achieved.

## What are Agents and Tools?

*   **Tool:** A tool is a function that performs a specific task. It could be anything from a web search engine to a calculator to a database lookup. In LangChain, a Tool has a name, a description, and a `run()` method that takes a string as input and returns a string as output.
*   **Agent:** An Agent is a chain powered by an LLM that decides which tool to use and in what order. You give the agent a set of tools and a goal, and the agent uses its reasoning abilities to figure out how to achieve that goal.

**Analogy: An Agent is like a project manager, and Tools are like team members.**
-   The **Agent (project manager)** receives a project goal (e.g., "Research the best hotels in Paris and book one for next week").
-   The **Agent** has access to a team of **Tools (team members)**:
    -   A "Web Search" tool for finding information online.
    -   A "Calculator" tool for performing calculations.
    -   A "Hotel Booking" tool for making reservations.
-   The **Agent** uses its reasoning abilities (powered by an LLM) to decide which tool to use and in what order to achieve the project goal. It might start by using the "Web Search" tool to find the best hotels, then use the "Hotel Booking" tool to make a reservation.

## Step-by-Step Code Tutorial

Let's build a simple agent that can use a web search tool to answer questions.

### 1. Install Necessary Libraries

We'll need `langchainhub` to access the search tool and `serpapi` to use the SerpAPI search engine. You'll also need a SerpAPI API key.

```bash
pip install langchainhub serpapi
```

### 2. Set the SerpAPI Key

Set the `SERPAPI_API_KEY` environment variable.

### 3. Create the Script

The `agent_tools_example.py` script demonstrates how to create and use an agent with tools.

### Key Concepts in the Code

1.  **`from langchain.agents import initialize_agent, Tool`**: We import the necessary modules for creating agents and defining tools.
2.  **`from langchainhub import serper`**: We import the SerpAPI tool from LangChain Hub.
3.  **`tools = [Tool(...)]`**: We define a list of tools that our agent can use. Each tool has a `name`, a `func` (the function to execute), and a `description`.
4.  **`agent = initialize_agent(...)`**: We create the agent.
    *   `llm`: The language model that will power the agent's reasoning.
    *   `tools`: The list of tools the agent can use.
    *   `agent="zero-shot-react-description"`: The type of agent to use. `zero-shot-react-description` is a common type that uses the tool descriptions to decide which tool to use.
    *   `verbose=True`: This is very useful for debugging. It prints out the agent's thought process at each step.

5.  **`agent.run(query)`**: We give the agent a query, and it uses its reasoning abilities and the available tools to try to answer the question.

## Why are Agents and Tools So Powerful?

*   **Beyond Pre-trained Knowledge:** Agents can access real-time information and external data sources, allowing them to answer questions that go beyond the LLM's training data.
*   **Automation:** Agents can automate complex tasks that would otherwise require manual intervention.
*   **Adaptability:** Agents can adapt to new situations and tasks by learning to use new tools.
*   **Reasoning and Planning:** Agents can break down complex problems into smaller steps and plan how to use the available tools to achieve the desired outcome.

## Conclusion of Part 2

Congratulations! You've now completed the second part of the LangChain tutorial. You've learned how to build intelligent applications that can:
*   Load data from various sources.
*   Split large documents into manageable chunks.
*   Embed text and store it in vector stores for efficient retrieval.
*   Use retrievers to find relevant information.
*   Remember past conversations with memory components.
*   Use agents and tools to automate complex tasks.

You now have the skills to build a wide range of sophisticated LLM applications.

**Next Up:** [Part 2 Project](./../project/README.md)
