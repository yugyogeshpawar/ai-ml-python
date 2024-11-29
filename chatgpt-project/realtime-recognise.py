import speech_recognition as sr
import openai
from gtts import gTTS
import pyttsx3


openai.api_key = ""


engine = pyttsx3.init()




def text_to_speech_google(mytext, language):
    audio_obj = gTTS(text=mytext, lang=language, slow=False)
    audio_obj.save("text_to_speech.mp3")


def chat_with_gpt(prompt):

    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(completion.choices[0].message.content)

    except Exception as e:
        print(e)


def real_time_recognization():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening....")
        while True:
            try:
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=5)
                result = recognizer.recognize_google(audio)
                chat_with_gpt(result)
            except Exception as e:
                print(f"An error occured: {e}")



def text_to_speech_pyttsx3(text):

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.save_to_file(text, 'test.mp3')
    engine.runAndWait()


text_to_speech_pyttsx3("Hello my name is John")





# text_to_speech_google("Hello my name is John", "en")




# real_time_recognization()



# requirments
# pip install SpeechRecognition
# pip install pyaudio
