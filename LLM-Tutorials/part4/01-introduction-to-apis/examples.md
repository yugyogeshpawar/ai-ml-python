# Examples in Action: APIs

APIs are the connective tissue of the internet. Here are some concrete examples of APIs you probably use every day without realizing it.

### 1. A Weather App

*   **The Goal:** You open a weather app on your phone to see the forecast for your city.
*   **The API Call:**
    1.  **Your Phone (The Client):** Your phone knows its current location (e.g., latitude and longitude). It sends a `GET` request to the weather service's API.
    2.  **The Endpoint:** The request goes to an endpoint like `https://api.weather.com/v3/forecast/daily?geocode=34.05,-118.24`.
    3.  **The API Key:** Your request includes an API key that identifies the weather app you're using.
    4.  **The Server:** The weather service's server receives the request, looks up the forecast data for those coordinates in its massive database, and formats it as a JSON response.
*   **The Result:** The server sends the JSON data back to your phone. The weather app doesn't show you the raw JSON; it *parses* that data and uses it to display the user-friendly icons and temperatures you see on the screen.

---

### 2. Embedded Google Maps on a Website

*   **The Goal:** A restaurant's website has a small, interactive map showing its location.
*   **The API Call:**
    1.  **The Website (The Client):** The website's code contains a small snippet that calls the Google Maps API.
    2.  **The Endpoint:** The request asks Google's servers for the map tiles corresponding to the restaurant's address.
    3.  **The Server:** Google's servers process this request and send back the visual map data.
*   **The Result:** The map you see on the website is not a static image. It's a live, interactive application running within the restaurant's webpage, powered by a constant stream of communication with Google's API. When you drag or zoom the map, the website is making new API calls to get the new map tiles.

---

### 3. "Log in with Google/Facebook" Buttons

*   **The Goal:** You sign up for a new service (e.g., Spotify) and you click "Log in with Google" to avoid creating a new password.
*   **The API Call (A simplified view using OAuth):**
    1.  **Spotify (The Client):** When you click the button, Spotify makes an API call to Google. The request says, "This user wants to log in. Please can you confirm their identity?"
    2.  **Google (The Server):** Google receives this request. Since you are already logged into Google in your browser, Google knows who you are. It asks you for permission: "Spotify wants to view your name and email address. Do you approve?"
    3.  **You:** You click "Approve."
    4.  **Google's Response:** Google's API sends a response back to Spotify's server. This response is a secure token that says, "We have successfully authenticated this user. Their name is John Doe and their email is john.doe@gmail.com."
*   **The Result:** Spotify's server receives this confirmation token. It now knows your identity without ever having to see your Google password. It creates a new Spotify account linked to your Google identity. This entire secure process is managed through APIs.

---

### 4. An LLM API Call

This is the example from our coding lesson.

*   **Goal:** Get a poem about a cat.
*   **The API Call:**
    1.  **Your Python Script (The Client):** Your script creates a connection to the OpenAI API using your secret key.
    2.  **The Request:** It sends a `POST` request to the `.../chat/completions` endpoint. The body of the request is a JSON object containing your prompt: `{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Write a poem about a cat"}]}`.
    3.  **OpenAI's Server:** The server receives the request, sends it to the GPT-3.5 Turbo model, gets the generated poem, and packages it into a JSON response.
*   **The Result:** Your script receives the JSON response, parses it to find the poem text inside `response.choices[0].message.content`, and prints it to your screen.
