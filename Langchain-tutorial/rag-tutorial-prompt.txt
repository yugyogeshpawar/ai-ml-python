# Prompt for Generating a Comprehensive RAG Tutorial

## 1. Overall Objective
Generate a complete, in-depth tutorial on building Retrieval-Augmented Generation (RAG) applications using LangChain. The tutorial should be suitable for hosting on GitHub and should cover everything from basic concepts to advanced, production-level RAG strategies.

## 2. Target Audience
The tutorial is for Python developers who have a basic understanding of LLMs and want to learn how to build applications that can reason over private or external data.

## 3. Core Philosophy & Style
- **Focus on the "Why":** Clearly explain *why* RAG is a crucial technique for building accurate and factual LLM applications (e.g., reducing hallucinations, using up-to-date information).
- **Modular Approach:** Break down the RAG pipeline into its distinct, logical steps (Load, Split, Embed, Store, Retrieve, Generate).
- **From Simple to Complex:** Start with a simple, end-to-end RAG chain and progressively introduce more advanced and optimized techniques.

## 4. High-Level Structure
The tutorial will be divided into three main parts.

- **Part 1: The Core RAG Pipeline**
- **Part 2: Advanced Retrieval Strategies**
- **Part 3: Production-Ready RAG**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following files inside a dedicated topic directory:
1.  **`README.md`**: The main lesson file with detailed explanations.
2.  **`[topic]_example.py`**: A runnable Python script.
3.  **`exercises.md`**: 2-3 practical exercises.
4.  **`interview_questions.md`**: 3 relevant interview questions and detailed answers.

---

### **Part 1: The Core RAG Pipeline**
- **Goal:** Build a foundational understanding of the end-to-end RAG process.
- **Topics:**
    1.  `01-introduction-to-rag`: What is RAG? Why is it important? Diagram of the full pipeline.
    2.  `02-loading-and-splitting`: Using `DocumentLoaders` and `TextSplitters` to prepare data.
    3.  `03-embeddings-and-vector-stores`: The concepts of semantic search, embeddings, and vector stores (`FAISS`, `Chroma`).
    4.  `04-basic-retrieval-and-generation`: Using `RetrievalQA` to build a simple, end-to-end RAG chain.

- **Project:** A simple Q&A script that answers questions about a single PDF document.

### **Part 2: Advanced Retrieval Strategies**
- **Goal:** Explore more sophisticated techniques to improve the quality of retrieval.
- **Topics:**
    1.  `01-retriever-configuration`: Configuring the retriever (`k`, search scores).
    2.  `02-metadata-filtering`: Using metadata to filter search results for more precise retrieval.
    3.  `03-mmr-and-diversity`: Using Maximum Marginal Relevance (MMR) to get more diverse and comprehensive results.
    4.  `04-contextual-compression`: Using `ContextualCompressionRetriever` to filter out irrelevant information from retrieved documents *before* sending them to the LLM.

- **Project:** A Q&A system that can answer questions over a collection of documents and allows the user to filter by source or category.

### **Part 3: Production-Ready RAG**
- **Goal:** Cover techniques for building robust, scalable, and evaluatable RAG systems.
- **Topics:**
    1.  `01-query-transformation`: Techniques like HyDE (Hypothetical Document Embeddings) to improve retrieval by transforming the user's query.
    2.  `02-reranking`: Using a second-stage model (like a cross-encoder) to re-rank the retrieved documents for better relevance.
    3.  `03-parent-document-retriever`: A strategy where you retrieve small chunks but provide the larger "parent" document to the LLM for better context.
    4.  `04-evaluation-with-ragas`: Introducing the Ragas framework for evaluating the performance of your RAG pipeline (e.g., faithfulness, answer relevancy).

- **Project:** A complete RAG pipeline deployed with LangServe that includes query transformation and a simple re-ranking step.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Clearly explain the role of each component in the RAG pipeline.
    - Use diagrams to illustrate data flow.
- **For `[topic]_example.py` files:**
    - Code should be runnable and well-commented.
- **For `exercises.md` and `interview_questions.md` files:**
    - Focus on the trade-offs between different RAG techniques (e.g., when to use MMR vs. standard search).
    - Questions should probe for a deep understanding of how to optimize and evaluate RAG systems.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
