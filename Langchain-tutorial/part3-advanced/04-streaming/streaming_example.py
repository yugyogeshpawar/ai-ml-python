from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. Create the LCEL chain (same as before)
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# 2. Use .stream() instead of .invoke()
# This returns a generator that yields chunks of the response as they are generated.
stream = chain.stream({"topic": "a programmer"})

# 3. Iterate over the stream and print each chunk
print("Streaming response:")
for chunk in stream:
    print(chunk, end="", flush=True)

print("\n\nStreaming complete.")
