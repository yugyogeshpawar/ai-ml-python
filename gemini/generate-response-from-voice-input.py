
import google.generativeai as genai
import speech_recognition as sr
from dotenv import load_dotenv
import os



load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


def chat_with_gemini(prompt):

    if prompt == "":
        print("Empty prompt")
        return

    # Configure your API key
    genai.configure(api_key=api_key) 

    # Set up the model
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Generate content
    response = model.generate_content(prompt)
    print("Gemini Response: ",response.text)


def real_time_recognization():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening....")
        while True:
            try:
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=5)
                result = recognizer.recognize_google(audio)
                print("Your prompt: ", result)
                chat_with_gemini(result)
            except Exception as e:
                print(f"An error occured: {e}")

real_time_recognization()
