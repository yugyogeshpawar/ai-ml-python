# 5. Memory: Giving Your Chains a Long-Term Memory

So far, our chains have been stateless. Every time we run a chain, it's a brand new interaction. It has no memory of previous conversations. This is a major limitation for building applications like chatbots, where context is key.

This is the problem that **Memory** components solve.

## What is Memory?

In LangChain, Memory is the mechanism that allows a chain or agent to remember previous interactions. It provides a way to read from and write to the conversation history, so that this history can be passed to the LLM in subsequent calls.

**How it works:**
1.  When a chain with memory is run, it first reads the conversation history from its memory component.
2.  It includes this history (along with the new user input) in the prompt that is sent to the LLM.
3.  After the LLM generates a response, the chain updates the memory with the new user input and the LLM's response.
4.  This updated history is then available for the next interaction.

LangChain provides several types of memory, from simple buffers that store the entire conversation to more complex summarization buffers that distill the conversation to save tokens.

## Step-by-Step Code Tutorial

Let's see how to add memory using the dedicated `ConversationChain` and `ConversationBufferMemory`.

### 1. Import the Necessary Components

We'll import `ConversationChain` and `ConversationBufferMemory`.

```python
from langchain.chains import ConversationChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
```

### 2. Create the Script

The `memory_example.py` script demonstrates how to build a conversational chain.

### Key Concepts in the Code

1.  **`memory = ConversationBufferMemory()`**: We create a simple instance of our memory component. The `ConversationChain` is smart enough to know how to use it without needing explicit `memory_key` variables.

2.  **`conversation_chain = ConversationChain(llm=llm, memory=memory)`**: We create our `ConversationChain`, passing in the `llm` and the `memory` object. The `ConversationChain` comes with a default prompt that is already designed for conversation, so we don't need to create our own `PromptTemplate`.

3.  **`conversation_chain.invoke(input="...")`**: We call the chain multiple times using `.invoke()`.
    *   **First call:** The memory is empty. The chain gets a response from the LLM and saves the "Human: Hi!" and the AI's response to the memory.
    *   **Second call:** The chain first loads the history from memory. It then includes this history in the new prompt, along with the new input ("I'm doing great! Just learning about LangChain."). The LLM now has the context of the previous turn and can generate a relevant response.

## Why is Memory So Important?

*   **Contextual Conversations:** Memory is the difference between a simple Q&A bot and a true conversational assistant. It allows for follow-up questions, clarifications, and a more natural flow of dialogue.
*   **Personalization:** By remembering user preferences or previous interactions, you can create a more personalized and helpful user experience.
*   **Complex Task Execution:** For agents that need to perform multi-step tasks, memory is essential for keeping track of what has been done and what needs to be done next.

## Next Steps

We've now covered the core components for building both data-aware (RAG) and conversational applications. The final and most advanced concept in this part of the tutorial is to combine these ideas.

What if we could build an application that not only remembers the conversation but can also decide for itself which tools (like our RAG retriever) to use to answer a question? This is the world of **Agents and Tools**.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Agents and Tools](./../06-agents-tools/README.md)
