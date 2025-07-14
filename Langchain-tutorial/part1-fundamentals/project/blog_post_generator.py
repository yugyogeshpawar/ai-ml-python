import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
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
chain = LLMChain(llm=llm, prompt=prompt_template)

# --- 5. Get User Input and Run the Chain ---
if __name__ == "__main__":
    # Check for API Key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set the key and run the script again.")
    else:
        topic = input("Enter a topic for the blog post: ")

        # Run the chain
        output = chain.run(topic)

        # Parse the output
        parsed_output = parser.parse(output)

        # Print the result
        print("\n--- Generated Blog Post ---\n")
        print(f"Title: {parsed_output.title}")
        print(f"Content:\n{parsed_output.content}")
