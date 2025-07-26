from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableMap
from langchain.memory.chat_message_histories import InMemoryChatMessageHistory

# 1. Define your schema
class ResumeInfo(BaseModel):
    total_experience: str = Field(description="Total years of experience")
    experience_in_ai_ml: str = Field(description="Years in AI/ML")
    location: str
    phone_number: str
    experience_in_python: str
    is_he_or_she_worked_on_ai_project: bool
    key_skills: list[str]
    graduation_done_in_relavent_to_ai_or_sortware_side: bool

# 2. Input resume content
resume_content = """
AI/ML Developer
"""

# 3. Output parser and formatting instructions
output_parser = JsonOutputParser(pydantic_object=ResumeInfo)
format_instructions = output_parser.get_format_instructions()

# 4. Prompt
prompt = ChatPromptTemplate.from_template(
    "Extract the following details from the resume:\n"
    "{format_instructions}\n\n"
    "Resume:\n{resume}"
).partial(format_instructions=format_instructions)

# 5. Model
model = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.0-flash")

# 6. Chain
chain = prompt | model | output_parser

# 7. Define a memory store
memory_store = {}

# 8. Use `RunnableWithMessageHistory` to enable memory
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: InMemoryChatMessageHistory(),
    input_messages_key="resume",  # this should match your input key
    history_messages_key="messages"
)

# 9. Simulate a conversation session
session_id = "user-session-001"

# 10. Invoke the chain with memory
result = chain_with_history.invoke(
    {"resume": resume_content},
    config={"configurable": {"session_id": session_id}}
)

# 11. Print
print("--- Original Resume ---")
print(resume_content)
print("\n--- Extracted Info ---")
print(result)
