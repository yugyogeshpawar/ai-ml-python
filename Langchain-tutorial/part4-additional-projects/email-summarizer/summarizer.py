from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. Create a sample email
email_content = """
Subject: Project Phoenix - Weekly Update

Hi Team,

This week, we've made significant progress on the new user authentication module. The front-end components are 80% complete, and the back-end API is fully functional. We encountered a minor setback with the database migration, which has caused a 2-day delay. We expect to resolve this by Wednesday.

Next week, we will focus on integrating the front-end and back-end, and we'll begin user acceptance testing (UAT). Please ensure all your code is committed by EOD Friday.

Action Items:
- Alice: Finalize the UI for the password reset page.
- Bob: Deploy the latest API version to the staging environment.
- Charlie: Prepare the UAT test cases.

Best,
Project Manager
"""

# 2. Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "Summarize the following email into 3-5 key bullet points:\n\n{email}"
)

# 3. Initialize the Chat Model
model = ChatOpenAI(temperature=0)

# 4. Create an output parser
output_parser = StrOutputParser()

# 5. Build the LCEL chain
chain = prompt | model | output_parser

# 6. Run the chain
summary = chain.invoke({"email": email_content})

# 7. Print the result
print("--- Original Email ---")
print(email_content)
print("\n--- Generated Summary ---")
print(summary)
