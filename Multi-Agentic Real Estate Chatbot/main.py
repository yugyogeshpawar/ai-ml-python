
import gemini_utils

gemini_prompt = input("Ask anything: ")
image_path = "images2.jpeg"

# gemini_utils.mygeminicall(gemini_prompt)


gemini_utils.geminicallwithimage(gemini_prompt, image_path)
