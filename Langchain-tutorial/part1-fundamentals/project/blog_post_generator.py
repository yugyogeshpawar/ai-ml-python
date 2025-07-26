import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- 1. Define the Output Structure ---
class BlogPost(BaseModel):
    title: str = Field(description="The title of the blog post")
    content: str = Field(description="The content of the blog post, written in a markdown format")

# --- 2. Set up the Parser ---
parser = PydanticOutputParser(pydantic_object=BlogPost)

# --- 3. Create the Prompt Template ---
prompt_template = PromptTemplate(
    template="Generate a short blog post about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# --- 4. Initialize the LLM and Chain ---
llm = OpenAI(temperature=0.7)
chain = prompt_template | llm | parser

# --- 5. Get User Input and Run the Chain ---
if __name__ == "__main__":
    # Check for API Key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set the key and run the script again.")
    else:
        topic = input("Enter a topic for the blog post: ")

        # Run the chain
        parsed_output = chain.invoke({"topic": topic})

        # Print the result
        print("\n--- Generated Blog Post ---\n")
        print(f"Title: {parsed_output.title}")
        print(f"Content:\n{parsed_output.content}")
