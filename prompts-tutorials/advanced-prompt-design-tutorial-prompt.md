# Prompt for Generating a Comprehensive Advanced Prompt Design Tutorial

## 1. Overall Objective
Generate a single, complete, multi-part tutorial on Advanced Prompt Design, suitable for hosting on GitHub. The tutorial should cover the entire lifecycle of modern prompting, from advanced engineering techniques to data-driven optimization and stateful, context-aware modeling.

## 2. Target Audience
The tutorial is for AI developers, advanced prompt engineers, and researchers who want to master the end-to-end process of building and managing sophisticated, production-level prompts and conversational systems. It assumes a solid understanding of basic prompt engineering.

## 3. Core Philosophy & Style
- **Integrated Approach:** Teach prompt design as a holistic discipline that combines the art of engineering, the science of optimization, and the architecture of context management.
- **Systematic and Rigorous:** Emphasize data-driven evaluation and systematic workflows over simple tricks.
- **Practical and Production-Oriented:** Focus on techniques and tools that are directly applicable to building robust, real-world AI applications.

## 4. High-Level Structure
The tutorial will be divided into three main parts, logically flowing from design to optimization to stateful implementation.

- **Part 1: Advanced Prompt Engineering Techniques**
- **Part 2: Prompt Optimization and Evaluation**
- **Part 3: Context-Aware and Stateful Modeling**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following files inside a dedicated topic directory:

1.  **`README.md`**: The main lesson file with detailed explanations, diagrams, and code examples.
2.  **`example.py`**: A runnable Python script demonstrating the concept (where applicable).
3.  **`prompt_examples.md`**: A file containing relevant prompt examples, conversation logs, or sample evaluation data.
4.  **`key_concepts.md`**: A file summarizing the key takeaways and vocabulary for the lesson.

---

### **Part 1: Advanced Prompt Engineering Techniques**
- **Goal:** Master sophisticated prompting strategies for complex reasoning and generation tasks.
- **Topics:**
    1.  `01-mastering-prompt-components`: A deep dive into the interplay of Role, Task, Context, Format, and Tone.
    2.  `02-chain-of-thought-and-self-consistency`: Using step-by-step reasoning and majority voting to improve accuracy on complex problems.
    3.  `03-structuring-prompts-for-rag`: Advanced techniques for structuring prompts that use externally retrieved documents.
    4.  `04-defensive-prompting-and-security`: Strategies to protect against prompt injection, data leakage, and other adversarial attacks.

- **Project:** Design a single, robust prompt that uses Chain-of-Thought and is designed to safely handle user-provided data in a RAG context.

### **Part 2: Prompt Optimization and Evaluation**
- **Goal:** Learn a data-driven workflow for systematically testing, refining, and compressing prompts.
- **Topics:**
    1.  `01-building-an-evaluation-framework`: How to create evaluation datasets and define metrics (e.g., semantic similarity, LLM-as-a-judge).
    2.  `02-manual-optimization-and-compression`: Heuristic techniques for shortening prompts while preserving performance.
    3.  `03-automated-prompt-optimization`: Using libraries (like `prompttools`) to programmatically test hundreds of prompt variations.
    4.  `04-cost-and-latency-analysis`: How to measure and optimize the financial and performance costs of your prompts.

- **Project:** Build a complete evaluation harness in Python that takes a list of prompt variations, runs them against a test dataset, and generates a report comparing their accuracy, cost, and latency.

### **Part 3: Context-Aware and Stateful Modeling**
- **Goal:** Build AI systems that can maintain and utilize context over long conversations.
- **Topics:**
    1.  `01-fundamentals-of-conversational-memory`: An overview of memory strategies, from simple buffers to summarization.
    2.  `02-advanced-memory-vector-stores`: Using a vector database as a long-term memory to retrieve relevant past interactions.
    3.  `03-stateful-agents-with-langgraph`: A practical guide to using a framework like LangGraph to manage state, context, and tool use in complex, cyclical agents.
    4.  `04-hybrid-context-systems`: Combining multiple memory types (e.g., a short-term buffer with long-term vector retrieval) for robust, multi-turn applications.

- **Project:** A stateful research agent that remembers the results of its previous actions to inform its next steps, effectively building a "mental model" of its progress on a complex task.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Use clear "Before" and "After" examples for optimization techniques.
    - Use diagrams to illustrate memory systems and agent state flows.
- **For `example.py` files:**
    - The code should be practical and clearly demonstrate the concept being taught.
- **For `prompt_examples.md` files:**
    - Provide diverse and well-annotated examples to illustrate the concepts.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
