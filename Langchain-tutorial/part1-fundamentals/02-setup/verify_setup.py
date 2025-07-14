import os
from langchain.llms import OpenAI

# Check if the API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY environment variable not set.")
else:
    # Initialize the OpenAI LLM
    llm = OpenAI()

    # Send a simple prompt to the LLM
    response = llm.predict("Tell me a joke.")

    # Print the response
    print(response)
