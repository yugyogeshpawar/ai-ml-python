# Exercises: Working with LLMs

These exercises will help you get comfortable with initializing and interacting with LLMs in LangChain.

### Exercise 1: Experiment with Temperature

1.  Copy the `llm_example.py` script from the lesson.
2.  Run the script three times with `temperature=0.0`. Note the responses. Are they the same?
3.  Now, run the script three times with `temperature=1.0`. Note the responses. How do they differ from each other and from the responses at `temperature=0.0`?
4.  Write a short comment in your script explaining what you observed about the `temperature` parameter.

### Exercise 2: Try a Different Model

1.  The default `OpenAI` model is `gpt-3.5-turbo-instruct`. LangChain allows you to specify different models.
2.  Modify your script to use a different model, such as `text-davinci-002`. You can do this by passing the `model_name` parameter during initialization: `llm = OpenAI(model_name="text-davinci-002")`.
3.  Run the script with the same prompt. Does the new model give a different style of response?

*(Note: Model availability and naming can change. If `text-davinci-002` is not available, you can find other model names in the OpenAI documentation.)*

### Exercise 3: Use a Chat Model

LangChain distinguishes between `LLMs` (text-in, text-out) and `ChatModels` (messages-in, message-out). Let's try a Chat Model.

1.  First, install the required package: `pip install langchain-openai`
2.  Modify your script to use `ChatOpenAI` instead of `OpenAI`.
3.  The input to a Chat Model is a list of messages. You'll need to import `HumanMessage` from `langchain.schema`.

Here's a code snippet to get you started:

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize the Chat Model
chat = ChatOpenAI(temperature=0.6)

# Create a list of messages
messages = [
    HumanMessage(content="What would be a good name for a company that makes colorful socks?")
]

# Get a prediction
response = chat.predict_messages(messages)

print(response.content)
```

Adapt the lesson's example to use this `ChatOpenAI` model and observe the difference in how you interact with it.
