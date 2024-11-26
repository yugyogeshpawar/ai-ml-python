import openai
from dotenv import load_dotenv
import os
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
load_dotenv()

openai.api_key = ""

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

# input_var = input("Enter prompt: ")

# chat_with_gpt(input_var)

def record():
    print("Recording....")
    audio_data = sd.rec(int(5*44100), 44100, 1, dtype='int16')
    sd.wait() 
    print("Recording finished.")
    write("output.wav", 44100, audio_data)
    print("File is saved")


def transcribe_audio():
    print("Calling transcribe")   
    model = whisper.load_model("base")
    print("Whisper model is loaded")
    result = model.transcribe("output.wav")
    print(result['text'])
    chat_with_gpt(result['text'])

# record()

transcribe_audio()


