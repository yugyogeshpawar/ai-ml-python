from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLM
llm = OpenAI(temperature=0.6)

# Define the Prompt Template
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Create the LLMChain. This chain will combine the LLM and the prompt.
name_chain = LLMChain(llm=llm, prompt=prompt_template)

# Define the product we want a name for
product_name = "artisanal coffee beans"

# Run the chain.
# The chain handles formatting the prompt with the input variable and calling the LLM.
response = name_chain.run(product_name)

print(f"Suggest a name for a company that makes {product_name}:")
print(f"Response: {response}")
