# 🧠 Python Assignment – Text to Speech (TTS) and Speech to Text (STT)

## 📘 What You Will Learn:

* How to make your Python programs **talk** using Text to Speech (TTS).
* How to make your programs **listen and understand** what you say using Speech to Text (STT).

---

## 🔊 **Text-to-Speech (TTS) Questions**

### 1. Convert Text to Speech (Basic)

Write a Python program that asks the user to type a sentence. Then, it reads the sentence aloud using the `pyttsx3` library.

### 2. Change the Voice Settings

Update your TTS program to change how the voice sounds. Try changing:

* The **speed** of the voice (faster or slower).
* The **volume** (louder or softer).
* The **voice type** (male or female, if available).

### 3. Save Speech as an Audio File

Write a program that takes a paragraph and converts it to an audio file (like `.mp3` or `.wav`). This way, the speech can be saved and played later.

### 4. Read Text from a File and Speak

Write a program that opens a `.txt` file, reads the content, and speaks it out loud.

### 5. Speak in Different Languages

Use the `gTTS` (Google Text-to-Speech) library to speak the same sentence in different languages (e.g., English and Hindi or any other language).

---

## 🗣️ **Speech-to-Text (STT) Questions**

### 6. Convert Speech to Text (Basic)

Use the `speech_recognition` library to listen to your voice using a microphone and print the text of what you said.

### 7. Convert Audio File to Text

Write a program that takes an audio file (like a `.wav` file) and converts the speech in the file into text.

### 8. Handle Errors in Speech Recognition

Update your STT program to handle problems such as:

* When no voice is detected.
* If the speech is unclear.
* If there is no internet (when using Google Speech API).

### 9. Voice-Controlled Commands

Create a simple voice-controlled app that listens to commands like:

* "Open Notepad"
* "Play music"
* "Close browser"
  And then performs the matching action using Python.

### 10. Combine TTS and STT

Make a full cycle program:

* Take a text input.
* Convert it to speech and play it.
* Record the spoken audio using the microphone.
* Convert the recorded speech back to text.
* Compare the original text and the final recognized text.

---

### ✅ Bonus Tips:

* Use libraries like `pyttsx3`, `gTTS`, `playsound`, and `speech_recognition`.
* Make sure your microphone and speakers are working.
* You might need internet for some features (like Google STT or gTTS).

