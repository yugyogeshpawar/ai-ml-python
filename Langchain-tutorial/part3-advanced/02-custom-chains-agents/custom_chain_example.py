from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. Define a custom function (Tool)
def check_domain(company_name: str) -> str:
    """
    This is a dummy function to check domain availability.
    In a real application, this would make an API call to a domain registrar.
    """
    # Placeholder logic:
    if len(company_name) > 15:
        return f"Error: {company_name} is too long (max 15 characters)."
    elif " " in company_name:
        return f"Error: {company_name} contains spaces."
    else:
        return f"{company_name}.com is available!"

# 2. Create a ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template("What is a good name for a company that makes {product}?")

# 3. Initialize the Chat Model
model = ChatOpenAI(temperature=0.7)

# 4. Create an Output Parser
output_parser = StrOutputParser()

# 5. Build the chain using LCEL
# The chain now represents the entire sequence of operations:
# prompt -> model -> output_parser -> check_domain
chain = prompt_template | model | output_parser | check_domain

# 6. Run the chain
response = chain.invoke({"product": "colorful socks"})

# Print the response
print(response)
