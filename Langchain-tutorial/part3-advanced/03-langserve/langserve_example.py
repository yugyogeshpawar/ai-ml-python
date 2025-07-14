from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import uvicorn

# 1. Create the FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 2. Create the LCEL chain
prompt = ChatPromptTemplate.from_template("What is a good name for a company that makes {product}?")
model = ChatOpenAI(temperature=0.7)
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# 3. Add the chain to the app
# This creates a /company-name endpoint for our chain.
add_routes(
    app,
    chain,
    path="/company-name",
)

# 4. Add a main block to run the server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
