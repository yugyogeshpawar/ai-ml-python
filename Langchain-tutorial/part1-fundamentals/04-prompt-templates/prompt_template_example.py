from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = OpenAI(temperature=0.6)

# Define the template. The curly braces {} denote a variable.
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Format the prompt by passing in a value for the 'product' variable.
# This creates the final prompt string.
formatted_prompt = prompt_template.format(product="eco-friendly water bottles")

# Get a prediction from the LLM using the formatted prompt
response = llm.invoke(formatted_prompt)

print(f"Prompt: {formatted_prompt}")
print(f"Response: {response}")
