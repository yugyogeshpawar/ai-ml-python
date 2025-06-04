
from google import genai

client = genai.Client(api_key="AIzaSyDL27GMn6-SO64ubxk7b0PuejazS6QemZg")


def mygeminicall(text):
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=text
    )
    print(response.text)


def geminicallwithimage(text, image_path):
    my_file = client.files.upload(file=image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, text],
    )
    print(response.text)


