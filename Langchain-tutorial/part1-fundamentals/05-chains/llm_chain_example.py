from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = OpenAI(temperature=0.6)

# Define the Prompt Template
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Create the chain by piping the prompt template to the LLM
name_chain = prompt_template | llm

# Define the product we want a name for
product_name = "artisanal coffee beans"

# Run the chain.
# The chain handles formatting the prompt with the input variable and calling the LLM.
response = name_chain.invoke({"product": product_name})

print(f"Suggest a name for a company that makes {product_name}:")
print(f"Response: {response}")
