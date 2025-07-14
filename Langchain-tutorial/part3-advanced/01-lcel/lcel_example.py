from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. Create a ChatPromptTemplate
# This is similar to a regular PromptTemplate, but optimized for Chat Models.
prompt = ChatPromptTemplate.from_template("What is a good name for a company that makes {product}?")

# 2. Initialize the Chat Model
# LCEL works best with Chat Models.
model = ChatOpenAI(temperature=0.7)

# 3. Create an Output Parser
# This simple parser converts the AIMessage content into a string.
output_parser = StrOutputParser()

# 4. Build the chain using LCEL
# The pipe operator (|) connects the components in a declarative way.
# The chain now represents the entire sequence of operations:
# prompt -> model -> output_parser
chain = prompt | model | output_parser

# 5. Run the chain
# The invoke() method takes a dictionary of inputs and returns the final output.
response = chain.invoke({"product": "colorful socks"})

# Print the response
print(response)
