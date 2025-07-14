# Prompt for Generating a Comprehensive LangChain Tutorial

## 1. Overall Objective
Generate a complete, multi-part LangChain tutorial suitable for hosting on GitHub. The tutorial should guide a user from a beginner level to an advanced level, covering the entire lifecycle of building and deploying LLM applications.

## 2. Target Audience
The tutorial is for Python developers who are new to LangChain but have some programming experience. It should be detailed enough for self-study and practical enough to prepare them for real-world projects and job interviews.

## 3. Core Philosophy & Style
- **Clarity and Detail:** Every concept must be explained clearly, avoiding jargon where possible or explaining it thoroughly when necessary. Use analogies to simplify complex topics.
- **Practicality:** Every lesson must be grounded in practical code examples that users can run themselves.
- **Completeness:** Each topic module must be self-contained and include theoretical explanations, code, exercises, and interview questions.
- **Structure:** The entire tutorial must follow a logical, hierarchical structure.

## 4. High-Level Structure
The tutorial will be divided into three main parts, plus a section for additional projects.

- **Part 1: The Fundamentals**
- **Part 2: Building Intelligent Applications**
- **Part 3: Advanced - Customization and Deployment**
- **Additional Projects**

## 5. Detailed Content Structure (For Each Topic)
For every topic within each part, you must generate the following four files inside a dedicated topic directory (e.g., `part1-fundamentals/01-introduction/`):

1.  **`README.md`**: The main lesson file.
2.  **`[topic]_example.py`**: A runnable Python script demonstrating the lesson's concept.
3.  **`exercises.md`**: A file containing 2-3 practical exercises.
4.  **`interview_questions.md`**: A file with 3 relevant interview questions and detailed answers.

---

### **Part 1: The Fundamentals**
- **Goal:** Introduce the core components of LangChain.
- **Topics:**
    1.  `01-introduction`: What is LangChain? Why use it? Core components. (No code example needed for this topic).
    2.  `02-setup`: Environment setup, `pip install`, and API key management.
    3.  `03-llms`: Using `LLM` and `ChatModel` classes, `temperature`.
    4.  `04-prompt-templates`: `PromptTemplate`, `ChatPromptTemplate`, f-strings vs. templates.
    5.  `05-chains`: `LLMChain`, the concept of chaining components.
    6.  `06-output-parsers`: `StrOutputParser`, `PydanticOutputParser`, structuring LLM output.
- **Project:** A command-line blog post generator that takes a topic and outputs a structured title and content.

### **Part 2: Building Intelligent Applications**
- **Goal:** Build applications that can reason and use external data (RAG).
- **Topics:**
    1.  `01-document-loaders`: `WebBaseLoader`, `PyPDFLoader`, the `Document` object.
    2.  `02-text-splitters`: The context window problem, `RecursiveCharacterTextSplitter`, `chunk_size`, `chunk_overlap`.
    3.  `03-embeddings-vector-stores`: What are embeddings? What are vector stores? `OpenAIEmbeddings`, `FAISS`.
    4.  `04-retrievers`: `vectorstore.as_retriever()`, `k`, search types like `MMR`.
    5.  `05-memory`: `ConversationBufferMemory`, making chains stateful.
    6.  `06-agents-tools`: The agent-as-reasoner concept, defining `Tool`s, `initialize_agent`.
- **Project:** A Q&A chatbot that answers questions based on a PDF document.

### **Part 3: Advanced - Customization and Deployment**
- **Goal:** Cover modern, production-level LangChain development.
- **Topics:**
    1.  `01-lcel`: LangChain Expression Language, the `|` (pipe) operator, `.invoke()`, `.stream()`.
    2.  `02-custom-chains-agents`: Creating custom functions as chain links, creating custom tools for agents.
    3.  `03-langserve`: Deploying chains as REST APIs using FastAPI and `add_routes`.
    4.  `04-streaming`: Using `.stream()` and `.astream()` for real-time responses.
- **Project:** A research agent that uses a web search tool and is deployed as a streaming API with LangServe.

### **Additional Projects**
- **Goal:** Provide more real-world, useful examples.
- **Projects:**
    1.  `sql-qa-agent`: An agent that connects to a SQL database and answers natural language questions.
    2.  `email-summarizer`: A chain that takes a long email and produces a bullet-point summary.
    3.  `code-documentation-generator`: A chain that generates a docstring for a given Python function.

---

## 6. Content Generation Guidelines

- **For `README.md` files:**
    - Start with a clear, conceptual explanation of the topic. Use analogies.
    - Provide a "Step-by-Step Code Tutorial" section that explains the key concepts in the accompanying example script.
    - Explain *why* the component is important and when to use it.
    - Conclude with a "Next Steps" section that links to the next lesson.

- **For `[topic]_example.py` files:**
    - The code must be clean, runnable, and well-commented.
    - Explain the purpose of key lines or blocks of code in the comments.
    - Ensure all necessary imports are included.

- **For `exercises.md` files:**
    - Create 2-3 exercises that allow the user to practice the concepts from the lesson.
    - The exercises should be hands-on and require writing or modifying code.
    - Include conceptual exercises where appropriate (e.g., "Design a custom agent for...").

- **For `interview_questions.md` files:**
    - Create 3 distinct questions that a developer might face in an interview regarding the topic.
    - Provide detailed, well-explained answers for each question. The answers should demonstrate a deep understanding of the concept.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
