from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = OpenAI(temperature=0)

# Create the memory object.  This stores the chat history.
# memory_key: This is the variable name that will store the chat history.
# This variable name *must* also be present in the prompt template.
memory = ConversationBufferMemory(memory_key="chat_history")

# Create a prompt template that includes a place for the chat history
# Notice the {chat_history} variable in the template.
template = """You are a friendly chatbot having a conversation with a human.

{chat_history}
Human: {input}
Chatbot:"""
prompt = PromptTemplate(
    input_variables=["chat_history", "input"], template=template
)

# Create the LLMChain
# Pass in the llm, the prompt, and the memory
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Start the conversation
# The first time, the chat_history is empty, so the LLM doesn't have any context.
# The chain automatically updates the memory after each call.
print(conversation_chain.predict(input="Hi!"))
print(conversation_chain.predict(input="I'm doing great! Just learning about LangChain."))
print(conversation_chain.predict(input="What is the capital of France?"))
print(conversation_chain.predict(input="What did I say my name was?"))
