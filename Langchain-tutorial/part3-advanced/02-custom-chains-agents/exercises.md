# Exercises: Custom Chains and Agents

These exercises will help you practice building your own specialized components in LangChain.

### Exercise 1: Create a Custom Output Parser

1.  Create a custom output parser that takes a string and returns a list of all the words in that string that are longer than 5 characters.
2.  Build an LCEL chain that:
    *   Takes a topic as input.
    *   Generates a short paragraph about that topic using an LLM.
    *   Uses your custom output parser to extract the long words from the LLM's response.
3.  Run the chain with a topic like "the history of the internet" and print the list of long words.

This exercise demonstrates how to create custom data transformation steps in your chains.

### Exercise 2: Build a Custom Tool for an Agent

1.  Create a function that takes a string as input and returns a "sentiment score" (e.g., "Positive", "Negative", "Neutral"). You can use a simple keyword-based approach for this (e.g., if the string contains "happy" or "good", return "Positive").
2.  Create a `Tool` object for this function, giving it a descriptive name and description.
3.  Create an agent that has access to this "Sentiment Analysis" tool and a "Web Search" tool.
4.  Give the agent a query that requires it to use your custom tool, such as: "What is the general sentiment of recent news about the stock market?"
5.  Observe how the agent first uses the "Web Search" tool to find news articles and then uses your "Sentiment Analysis" tool to analyze the results.

This exercise shows how to create custom tools that can be used by agents to perform specialized tasks.

### Exercise 3: Design a Custom Agent (Conceptual)

Imagine you want to build an agent that can help you plan a trip.

1.  **Define the Tools:** What tools would this agent need? Think about things like:
    *   `search_flights(origin, destination, date)`
    *   `search_hotels(city, check_in_date, check_out_date)`
    *   `get_weather_forecast(city, date)`
    *   `book_flight(flight_id)`
    *   `book_hotel(hotel_id)`
2.  **Design the Prompt:** What instructions would you give the agent in its prompt? How would you describe the tools so that the agent knows when to use them?
3.  **Plan the Execution:** If you gave the agent the query "Plan a 3-day trip to Paris for next month, including flights and a hotel," what steps would you expect the agent to take? How would it use the tools in sequence?

This exercise helps you think through the process of designing a custom agent from scratch, which is a key skill for building advanced LLM applications.
