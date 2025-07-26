# Try It Yourself: Interacting with a RAG System

Building a full RAG system from scratch is a complex coding task, but you can get a feel for how it works by interacting with public-facing applications that are built on this technology.

---

### The Scenario

We will use a tool called **Perplexity**. Perplexity is a conversational AI search engine that uses RAG as its core principle. When you ask it a question, it doesn't just answer from its internal knowledge. It first **retrieves** relevant web pages and then **generates** an answer based on the content of those pages, providing citations.

### Exercise: The "Open-Book" Search Engine

1.  **Go to Perplexity's website:**
    *   [https://www.perplexity.ai/](https://www.perplexity.ai/)

2.  **Ask a question about a very recent event.** This is something a standard, "closed-book" chatbot would fail at. Try a question like:
    > "What were the main announcements from Apple's most recent keynote event?"
    
    (If no event happened recently, you can ask about the latest major movie release or a recent sporting event.)

3.  **Analyze the response. Look for the RAG pattern:**
    *   **The Generation:** First, read the answer it provides. It should be a concise, well-written summary.
    *   **The Retrieval:** Now, look closely at the "Sources" or numbered citations it provides alongside the answer. These are the web pages it "retrieved" before generating the summary.
    *   **The Augmentation:** You can imagine the prompt that Perplexity constructed for its internal LLM: "Using the content from these 5 sources, please answer the user's question about Apple's recent keynote."

4.  **Verify the sources.** Click on one of the source links. Does the information in the source match the information in the AI's generated answer? This ability to check the sources is a key advantage of RAG systems and builds trust in the output.

5.  **Ask a question about a niche or specialized topic.** Try something that a general model might not know much about.
    > "What are the best practices for training a dog to compete in sheepdog trials?"

    Again, notice how it finds and cites specialized articles, blog posts, or forum discussions on this topic. It's augmenting its general knowledge with expert information it finds in real-time.

6.  **Compare with a standard chatbot.** Now, go to a standard chatbot that does *not* use live web search (you can often select a "focus" or mode in chatbots; choose one that isn't for web search, or use an older model if available). Ask the same two questions.
    *   For the recent event, the chatbot will likely tell you its knowledge is cut off in a certain year and it doesn't know about recent events.
    *   For the niche topic, it might give a very generic answer, lacking the specific details that Perplexity found in its retrieved sources.

**Reflection:**
This exercise makes the concept of RAG tangible. You can see the two-step process in action: the system retrieves sources (the "R" in RAG) and then uses them to generate a new answer (the "G"). This is why RAG is so powerful for creating applications that need to be accurate, up-to-date, and trustworthy.
