from langchain_openai import OpenAI

# Initialize the LLM with a temperature of 0.6
# Temperature controls the "creativity" or randomness of the output.
# A value of 0.0 is deterministic, while 1.0 is very creative.
llm = OpenAI(temperature=0.6)

# Define the prompt
prompt = "What would be a good name for a company that makes colorful socks?"

# Get a prediction from the LLM
# The .invoke() method sends the prompt to the model and returns the string response.
response = llm.invoke(prompt)

# Print the response
print(f"Prompt: {prompt}")
print(f"Response: {response}")
