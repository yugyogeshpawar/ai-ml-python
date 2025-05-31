from google import genai

client = genai.Client(api_key="<Key>")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="I can see cracks on my wall. How i can deal with it. tell me in short",
)

print(response.text)
