from langchain.chains import ConversationChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory

# Initialize the LLM
llm = OpenAI(temperature=0)

# Create the memory object.  This stores the chat history.
memory = ConversationBufferMemory()

# Create the ConversationChain
# Pass in the llm and the memory
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory
)

# Start the conversation
# The first time, the chat_history is empty, so the LLM doesn't have any context.
# The chain automatically updates the memory after each call.
print(conversation_chain.invoke(input="Hi!"))
print(conversation_chain.invoke(input="I'm doing great! Just learning about LangChain."))
print(conversation_chain.invoke(input="What is the capital of France?"))
print(conversation_chain.invoke(input="What did I say my name was?"))
