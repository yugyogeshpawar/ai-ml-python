from google.cloud import texttospeech

def text_to_speech_google_cloud(mytext, language_code="en-US", voice_name="en-US-Wavenet-D", speaking_rate=1.0):
    # Initialize the Text-to-Speech client
    client = texttospeech.TextToSpeechClient()

    # Set up the text input
    synthesis_input = texttospeech.SynthesisInput(text=mytext)

    # Set up the voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )

    # Set up the audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the audio to a file
    with open("realistic_voice.mp3", "wb") as out:
        out.write(response.audio_content)
        print("Audio content written to file 'realistic_voice.mp3'.")

# Example usage
text = "Hello, Yogesh! This is a natural-sounding text-to-speech voice."
text_to_speech_google_cloud(mytext=text)



# pip install google-cloud-texttospeech

# export GOOGLE_APPLICATION_CREDENTIALS="path_to_your_service_account.json"
