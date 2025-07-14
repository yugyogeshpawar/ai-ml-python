# Exercises: Agents and Tools

These exercises will help you explore the power and flexibility of agents and tools.

### Exercise 1: Add a Calculator Tool

1.  Copy the `agent_tools_example.py` script.
2.  In addition to the "Web Search" tool, add a "Calculator" tool to the `tools` list. You can use the `Tool` class directly with a simple Python function.
    ```python
    def calculate(expression: str) -> str:
        """Useful for getting the result of a math expression. The input should be a valid Python expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    tools = [
        Tool(
            name='Calculator',
            func=calculate,
            description="Useful for getting the result of a math expression. The input should be a valid Python expression."
        ),
        Tool(
            name="Web Search",
            func=serper.run,
            description="Useful for searching the web and getting concise answers to factual questions.",
        )
    ]
    ```
3.  Modify the query to something that requires both web search and calculation: "What is the current temperature in London, and what is that temperature in Fahrenheit?"
4.  Run the script and observe how the agent uses both tools to answer the question.

This exercise demonstrates how to combine multiple tools to solve more complex problems.

### Exercise 2: Create a Custom Tool

Let's create a tool that interacts with a simple API.

1.  Imagine you have a function that retrieves the current stock price for a given ticker symbol:
    ```python
    def get_stock_price(ticker: str) -> str:
        """Useful for getting the current stock price of a company."""
        # In a real application, this would make an API call to a stock data provider.
        # For this exercise, we'll just return a dummy value.
        if ticker == "AAPL":
            return "AAPL: $175.00"
        elif ticker == "GOOG":
            return "GOOG: $2700.00"
        else:
            return f"Could not find stock price for {ticker}"
    ```
2.  Create a `Tool` object for this function, giving it a descriptive name and description.
3.  Add this tool to the `tools` list in your `agent_tools_example.py` script.
4.  Modify the query to something like: "What is the current stock price of Apple (AAPL)?"
5.  Run the script and observe how the agent uses your custom tool to answer the question.

This exercise shows how to integrate your own code and APIs into a LangChain agent.

### Exercise 3: Add Memory to an Agent

1.  Combine concepts from this lesson and the previous one on Memory.
2.  Add `ConversationBufferMemory` to your agent.
3.  Modify the agent initialization to include the memory:
    ```python
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        memory=memory, # Add the memory here
        verbose=True,
    )
    ```
4.  Now, have a multi-turn conversation with the agent. Ask it a question that requires it to use a tool, then ask a follow-up question that refers to the previous answer. Does the agent remember the previous interaction?

This exercise demonstrates how to build a conversational agent that can access external knowledge and remember past interactions.
