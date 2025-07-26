# Examples in Action: AI Agents

AI agents are systems that can reason, plan, and use tools to accomplish a goal. While truly autonomous agents are still an emerging technology, many modern AI applications incorporate "agent-like" behaviors.

### 1. Complex Web Research

*   **The Goal:** "Create a detailed report on the market trends for electric vehicles in Europe for 2023, including key players, sales figures, and predictions for 2024."
*   **A Simple Chatbot's Approach:** A standard chatbot might give a generic summary based on its outdated training data. A RAG-based chatbot (like Perplexity) would do a single web search and summarize the top few results.
*   **An AI Agent's Approach (The ReAct Loop):**
    1.  **Reason:** "My goal is a detailed report. I need to break this down. First, I'll find the overall sales figures for EVs in Europe in 2023."
    2.  **Act:** Use `web_search` tool with query: "EV sales figures Europe 2023".
    3.  **Observe:** The search results show reports from several automotive agencies. It gets the total sales number.
    4.  **Reason:** "Okay, I have the total number. Now I need to identify the key players. I'll search for the best-selling EV models in Europe for that year."
    5.  **Act:** Use `web_search` tool with query: "best selling electric car models Europe 2023".
    6.  **Observe:** The results list models like the Tesla Model Y, VW ID.4, etc.
    7.  **Reason:** "Now I have the top models. I need predictions for 2024. I will search for analyst reports on this topic."
    8.  **Act:** Use `web_search` tool with query: "EV market forecast Europe 2024".
    9.  **Observe:** It finds several analyst reports.
    10. **Reason:** "I have all the pieces of information: total sales, key players, and future predictions. I can now synthesize this into the final report."
    11. **Act:** Generate the final, structured report.
*   **Result:** A comprehensive report built from multiple, targeted web searches, far more detailed than a single query could produce.

---

### 2. Automated Travel Booking

*   **The Goal:** "Book me a flight and hotel for a trip to London from New York, leaving next Monday and returning Friday. I want a non-stop flight and a 4-star hotel near the British Museum. My budget is $2000."
*   **An AI Agent's Approach:**
    1.  **Reason:** "I need to book a flight and a hotel with several constraints. First, I'll check for flights."
    2.  **Act:** Use a `flight_search` tool (an API for a service like Kayak or Google Flights) with parameters: `origin=JFK`, `destination=LHR`, `departure_date=...`, `return_date=...`, `non-stop=true`.
    3.  **Observe:** The tool returns a list of flights and their prices. The agent sees that the cheapest non-stop flight is $900.
    4.  **Reason:** "The flight is $900. This leaves $1100 for the hotel for 4 nights ($275/night). Now I need to find a hotel."
    5.  **Act:** Use a `hotel_search` tool with parameters: `location=British Museum`, `check_in=...`, `check_out=...`, `stars=4`, `max_price=275`.
    6.  **Observe:** The tool returns a few hotels that match the criteria.
    7.  **Reason:** "I have found a valid flight and hotel that meet the budget. I should now ask the user for confirmation before booking."
    8.  **Act:** Use the `ask_user` tool: "I've found a non-stop flight for $900 and the 'Montague on the Gardens' hotel for $260/night. The total is $1940. Shall I proceed with the booking?"
*   **Result:** The agent has handled a complex, multi-step transaction that involves searching multiple data sources, performing calculations, and interacting with the user for confirmation.

---

### 3. Automated Data Analysis

*   **The Goal:** "Here is a CSV file of our recent sales data. Can you tell me which product category had the highest growth last month and create a bar chart to visualize it?"
*   **An AI Agent's Approach (using a Code Interpreter tool):**
    1.  **Reason:** "The user has given me a file and a two-part question. First, I need to inspect the data to understand its structure."
    2.  **Act:** Use the `code_interpreter` tool to run Python code: `import pandas as pd; df = pd.read_csv('sales_data.csv'); print(df.head())`.
    3.  **Observe:** The agent sees the first few rows and the column names (e.g., `Date`, `Product`, `Category`, `Sales`).
    4.  **Reason:** "Okay, I see the structure. To find the growth, I need to group the data by 'Category' and sum the 'Sales' for this month and last month."
    5.  **Act:** Use the `code_interpreter` to run more pandas code to perform the calculation.
    6.  **Observe:** The code output shows that the "Electronics" category had the highest growth.
    7.  **Reason:** "I have the answer to the first part. Now I need to create a bar chart."
    8.  **Act:** Use the `code_interpreter` to run Python plotting code (using a library like Matplotlib or Seaborn) to generate the bar chart and save it as an image file.
*   **Result:** The agent provides a natural language answer ("The 'Electronics' category had the highest growth") and also provides the bar chart image it generated, having used its code tool to perform a complex data analysis workflow.
