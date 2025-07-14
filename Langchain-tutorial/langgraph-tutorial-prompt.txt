# Prompt for Generating a Comprehensive LangGraph Tutorial

## 1. Overall Objective
Generate a complete, multi-part LangGraph tutorial suitable for hosting on GitHub. The tutorial should explain how to build robust, stateful, multi-agent applications using LangGraph, assuming the user has a foundational understanding of LangChain.

## 2. Target Audience
The tutorial is for Python developers who are already familiar with the basics of LangChain (LLMs, Chains, Agents) and want to learn how to build more complex, cyclical, and resilient applications.

## 3. Core Philosophy & Style
- **Focus on State and Cycles:** Emphasize that LangGraph is for building applications with complex, cyclical flows, unlike the linear Directed Acyclic Graphs (DAGs) of standard chains.
- **Graph-Based Thinking:** Encourage a mental model of nodes and edges. Use analogies like flowcharts or state machines.
- **Practical, Agentic Examples:** The projects should showcase LangGraph's strengths, such as building multi-agent systems or chains that can dynamically modify their own structure.

## 4. High-Level Structure
The tutorial will be divided into two main parts, plus a section for advanced projects.

- **Part 1: The Fundamentals of LangGraph**
- **Part 2: Building Agentic Applications with LangGraph**
- **Advanced LangGraph Projects**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following files inside a dedicated topic directory:
1.  **`README.md`**: The main lesson file with detailed explanations.
2.  **`[topic]_example.py`**: A runnable Python script.
3.  **`exercises.md`**: 2-3 practical exercises.
4.  **`interview_questions.md`**: 3 relevant interview questions and detailed answers.

---

### **Part 1: The Fundamentals of LangGraph**
- **Goal:** Introduce the core concepts of building and running graphs.
- **Topics:**
    1.  `01-introduction`: What is LangGraph? Why use it over standard LCEL? The concept of stateful graphs.
    2.  `02-graph-state`: Defining a `State` object (e.g., using `TypedDict`) to pass information between nodes.
    3.  `03-nodes-and-edges`: Creating nodes (functions or runnables), adding nodes to the `Graph`, and defining the edges that connect them.
    4.  `04-conditional-edges`: The core of LangGraph's power. Creating edges that route to different nodes based on the current state.
    5.  `05-compiling-and-running`: Using `graph.compile()` to create a runnable graph and `.invoke()` to run it.

- **Project:** A "Reflective RAG" application. The graph will first retrieve documents. A second node will decide if the documents are relevant enough to answer the question. If yes, it generates an answer. If no, it triggers a web search to find more information and then re-evaluates.

### **Part 2: Building Agentic Applications with LangGraph**
- **Goal:** Create complex, multi-agent systems.
- **Topics:**
    1.  `01-single-agent-as-graph`: Rebuilding a standard LangChain agent (like a ReAct agent) as a LangGraph graph to understand the underlying mechanics.
    2.  `02-multi-agent-collaboration`: Creating a graph with multiple agents (nodes) that can work together. Example: a "Researcher" agent and a "Writer" agent.
    3.  `03-human-in-the-loop`: Adding a "wait" state to the graph that requires human input before continuing.
    4.  `04-managing-agent-state`: Techniques for updating and passing complex state between agentic nodes.
    5.  `05-tool-calling-in-graphs`: How to effectively manage tool execution and observation within a graph's state.

- **Project:** A multi-agent "Hierarchical Research Team." A "Chief" agent breaks down a complex research question into sub-tasks. It then dispatches these tasks to two "Worker" agents who can use tools (e.g., web search). The workers return their findings to the Chief, who then synthesizes the final answer.

### **Advanced LangGraph Projects**
- **Goal:** Provide more complex, real-world examples.
- **Projects:**
    1.  `self-correcting-rag`: A RAG system that grades its own retrieved documents and its final answer. If the quality is low, it can loop back to refine the query or search for more documents.
    2.  `dynamic-tool-selection-agent`: An agent that can decide *which set of tools* to use based on the user's initial query, effectively loading different "toolbelts" for different tasks.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Explain concepts from a graph-based perspective (nodes, edges, state).
    - Use diagrams (e.g., ASCII art) to illustrate the graph's structure.
- **For `[topic]_example.py` files:**
    - Code should clearly define the state, nodes, and graph construction.
    - Use `verbose=True` where possible to show the agent's thought process.
- **For `exercises.md` and `interview_questions.md` files:**
    - Follow the same high-quality standards as the LangChain tutorial prompt. Focus on state management, conditional logic, and agent collaboration.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
