# Key Concepts: AI Agents

Here are the key terms for understanding AI agents.

### 1. AI Agent
-   **What it is:** An AI system that can autonomously pursue a goal by reasoning, making plans, and using tools.
-   **Analogy:** A smart personal assistant. You don't tell them every single step. You give them a high-level goal (e.g., "Plan my weekend trip"), and they figure out the necessary sub-tasks (book hotel, find flights, check weather) and execute them on their own.
-   **Why it matters:** Agents represent a shift from **reactive AI** (which responds to commands) to **proactive AI** (which pursues goals). This is a major step towards more capable and autonomous systems.

### 2. Tool Use
-   **What it is:** The ability of an AI agent to use external tools to gather information or perform actions that it can't do on its own.
-   **Analogy:** A carpenter's toolbox. A carpenter can't build a house with their bare hands; they need a hammer, a saw, and a measuring tape. For an AI agent, tools are things like a web search API, a calculator, or a code interpreter.
-   **Why it matters:** Tool use overcomes the inherent limitations of an LLM. An LLM can't browse the live internet or do perfect math, but it can be the "brain" that decides *when* to use a tool that can.

### 3. ReAct (Reason + Act)
-   **What it is:** The core decision-making loop that most AI agents use. It's a cycle of reasoning about a problem and then taking an action (using a tool) to move closer to the solution.
-   **Analogy:** The scientific method. A scientist has a question (**Goal**). They form a hypothesis (**Reason**). They run an experiment to test it (**Act**). They look at the results (**Observe**). Based on the results, they form a new hypothesis and repeat the cycle.
-   **The Loop:**
    1.  **Reason:** The LLM thinks about what to do next.
    2.  **Act:** The agent chooses and uses a tool.
    3.  **Observe:** The agent gets the result from the tool.
    4.  **Repeat:** The result is fed back into the LLM to start the next reasoning step.
-   **Why it matters:** This simple loop is what allows agents to break down complex, long-term goals into a series of manageable, concrete actions.
