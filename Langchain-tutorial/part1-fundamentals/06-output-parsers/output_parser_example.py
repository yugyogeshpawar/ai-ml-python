from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

# Initialize the LLM
llm = OpenAI(temperature=0.6)

# Create an instance of the output parser
output_parser = CommaSeparatedListOutputParser()

# Get the format instructions from the parser
# This is a string that tells the LLM how to format its output
format_instructions = output_parser.get_format_instructions()

# Create the prompt template, including the format instructions
prompt_template = PromptTemplate(
    template="Provide 5 examples of {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

# Create the chain
chain = prompt_template | llm | output_parser

# Invoke the chain with a subject
parsed_output = chain.invoke({"subject": "popular dog breeds"})

print(f"Subject: popular dog breeds")
print(f"Parsed Response:\n{parsed_output}")
