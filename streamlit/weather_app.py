import streamlit as st

import requests


st.set_page_config(layout="wide")


st.subheader("My weather app 🌦️🌤️🤍☁️🌿🍃✨️🌧️")

API_KEY = "43b6f47b478c2eb4e60f1bcf00f874cf"

URL = "http://api.openweathermap.org/data/2.5/weather?"


city = st.text_input("Enter city name.")

if city:
    COMPLETE_URL = URL + "appid=" + API_KEY + "&q=" + city
    response = requests.get(COMPLETE_URL)
    data = response.json()
    print(response.json())
    if data["cod"] == 200:

        BG_URL = ""
        if data["weather"][0]["main"] == "Clouds":
            BG_URL = "https://media.istockphoto.com/id/186849963/photo/sky.jpg?s=612x612&w=0&k=20&c=qQHaTbThki442O54f2CPljWrYuq8QYL3qRJGwvIkCRg="
        elif data["weather"][0]["main"] == "Rain":
            BG_URL = "https://media.istockphoto.com/id/1476190237/photo/summer-rain-raindrops-bad-weather-depression.jpg?s=612x612&w=0&k=20&c=GUJvhhU3WVvHhJ3kU4E33fVUzICegLq1sh2msWS5kNk="
        elif data["weather"][0]["main"] == "Snow":
            BG_URL = "https://i.ytimg.com/vi/r6VkCdQQdG0/maxresdefault.jpg"
        # Then inject custom CSS
        page_bg_img = f'''
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("{BG_URL}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);  /* Transparent header */
        }}

        [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.5);  /* Optional: translucent sidebar */
        }}
        </style>
        '''

        # Inject CSS after setting page config
        st.markdown(page_bg_img, unsafe_allow_html=True)

        st.subheader(f"Weather in {data["name"]}")
        st.write(f"Temperature: {data["main"]["temp"]}")
        st.write(f"Humidity: {data["main"]["humidity"]}")
    else:
        st.error((data['message']))

