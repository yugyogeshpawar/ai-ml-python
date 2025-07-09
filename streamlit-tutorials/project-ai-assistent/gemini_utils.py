from google.genai import types
from google import genai

client = genai.Client(api_key="AIzaSyDOhlcF6po_OI1mYhCGX9a7fuHaUOJ6uOU")


def call_gemini(text, system_instructions):

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents = text,
        config=types.GenerateContentConfig(system_instruction=system_instructions),
        )
    return response
