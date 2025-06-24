import streamlit as st
from PyPDF2 import PdfReader
from google import genai
from google.genai import types


client = genai.Client(api_key="AIzaSyDOhlcF6po_OI1mYhCGX9a7fuHaUOJ6uOU")


st.write("Ai interviewer")

pdf_file = st.file_uploader("Upload your resume")

if pdf_file:

    reader = PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    text = ""
    for x in range(0, number_of_pages):
        page = reader.pages[x]
        text += page.extract_text()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents = text,
        config=types.GenerateContentConfig(system_instruction="You are a ai interviewer, first you have provided the extracted data of pdf now first orginize the data and give back."),
        )
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents="How does AI work?",
    #     config=types.GenerateContentConfig(
    #         thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    #     ),
    # )
    st.write(response.text)
