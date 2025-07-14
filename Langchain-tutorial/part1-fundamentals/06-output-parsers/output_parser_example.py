from langchain.llms import OpenAI
from langchain import PromptTemplate
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

# Format the prompt with a subject
formatted_prompt = prompt_template.format(subject="popular dog breeds")

# Get the response from the LLM
response = llm.predict(formatted_prompt)

# Parse the raw string output into a Python list
parsed_output = output_parser.parse(response)

print(f"Subject: popular dog breeds")
print(f"Raw Response:\n{response}")
print(f"Parsed Response:\n{parsed_output}")
