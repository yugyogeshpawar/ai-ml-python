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

Let's see how to add memory to an `LLMChain` using `ConversationBufferMemory`.

### 1. Import the Necessary Components

We need our `LLMChain` and `OpenAI` model, but now we also import `ConversationBufferMemory`.

```python
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
```

### 2. Create the Script

The `memory_example.py` script demonstrates how to build a conversational chain.

### Key Concepts in the Code

1.  **`memory = ConversationBufferMemory(...)`**: We create an instance of our memory component.
    *   `memory_key="chat_history"`: This is a crucial parameter. It's the name of the variable where the conversation history will be stored. This key **must** also be present in our prompt template's `input_variables`.

2.  **`template = """..."""`**: We define our prompt template. Notice that it now includes a placeholder for our memory variable: `{chat_history}`.

3.  **`prompt = PromptTemplate(...)`**: We create our `PromptTemplate`. Critically, we include `"chat_history"` in the `input_variables` list.

4.  **`conversation_chain = LLMChain(...)`**: We create our `LLMChain` as usual, but now we pass in the `memory` object we created.

5.  **`conversation_chain.predict(input="...")`**: We call the chain multiple times.
    *   **First call:** The `{chat_history}` variable is empty. The chain gets a response from the LLM and saves the "Human: Hi!" and the AI's response to the memory.
    *   **Second call:** The chain first loads the history from memory (`"Human: Hi!\nAI: Hello! How can I help you today?"`). It then includes this history in the new prompt, along with the new input ("I'm doing great! Just learning about LangChain."). The LLM now has the context of the previous turn and can generate a relevant response.

## Why is Memory So Important?

*   **Contextual Conversations:** Memory is the difference between a simple Q&A bot and a true conversational assistant. It allows for follow-up questions, clarifications, and a more natural flow of dialogue.
*   **Personalization:** By remembering user preferences or previous interactions, you can create a more personalized and helpful user experience.
*   **Complex Task Execution:** For agents that need to perform multi-step tasks, memory is essential for keeping track of what has been done and what needs to be done next.

## Next Steps

We've now covered the core components for building both data-aware (RAG) and conversational applications. The final and most advanced concept in this part of the tutorial is to combine these ideas.

What if we could build an application that not only remembers the conversation but can also decide for itself which tools (like our RAG retriever) to use to answer a question? This is the world of **Agents and Tools**.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Agents and Tools](./../06-agents-tools/README.md)
