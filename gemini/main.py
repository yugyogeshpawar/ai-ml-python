

import google.generativeai as genai

# Configure your API key
genai.configure(api_key="Your-api-key") #replace with your api key.

# Set up the model
model = genai.GenerativeModel('gemini-2.0-flash')

myinput = str(input("Enter you prompt: "))

# Generate content
response = model.generate_content(myinput)
print(response.text)
