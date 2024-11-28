import speech_recognition as sr

def real_time_recognization():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening....")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                result = recognizer.recognize_google(audio)
                print(f"You said: {result}")
            except Exception as e:
                print(f"An error occured: {e}")


real_time_recognization()



# requirments
# pip install SpeechRecognition
# pip install pyaudio