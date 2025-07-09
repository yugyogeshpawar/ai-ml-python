# Streamlit Tutorial: Building Web Apps with Python

Welcome to this beginner-friendly tutorial on building interactive web applications using Streamlit! Streamlit is an open-source Python library that makes it incredibly easy to create and share beautiful, custom web apps for machine learning and data science.

Whether you're a data scientist, engineer, or just someone who wants to share their Python scripts as interactive web tools, Streamlit is a fantastic choice.

Let's get started!

## 📘 Part 1: Introduction

### What is Streamlit?

Streamlit is a Python library that turns data scripts into shareable web apps in minutes. It's designed for data scientists and engineers who don't have much experience with web development frameworks like Flask or Django. With Streamlit, you write pure Python code, and it handles the web magic for you.

### Why use Streamlit?

*   **Speed:** Build and deploy apps incredibly fast.
*   **Simplicity:** Write apps purely in Python, no HTML, CSS, or JavaScript required.
*   **Interactive:** Easily add sliders, buttons, text inputs, and other widgets to make your apps interactive.
*   **Data-focused:** Great support for displaying data, charts, and media.
*   **Open Source:** Free to use and a growing community.

### How to install it?

Installing Streamlit is straightforward using pip, Python's package installer.

Open your terminal or command prompt and run the following command:

```bash
pip install streamlit
```

This command downloads and installs the Streamlit library and its dependencies.

### How it works (basic architecture)

Streamlit apps work differently from traditional web frameworks. When you run a Streamlit script:

1.  Streamlit starts a web server.
2.  When a user opens the app in their browser, Streamlit runs your Python script from top to bottom.
3.  Every time a user interacts with a widget (like moving a slider or clicking a button), Streamlit re-runs the entire script from top to bottom.
4.  Streamlit keeps track of variable states across these re-runs using a clever caching mechanism (`st.cache_data`, `st.cache_resource`, `st.session_state`).

This re-run behavior simplifies the code structure significantly compared to request-response models.

### Setting up the first Streamlit app (hello_world.py)

Let's create our first Streamlit application.

1.  Open your favorite code editor.
2.  Create a new Python file named `hello_world.py`.
3.  Add the following code:

```python
# Import the streamlit library
import streamlit as st

# Add a title to the app
st.title('Hello, Streamlit!')

# Add some text
st.write('This is my first Streamlit web app.')

# Add a header
st.header('Getting Started')

# Add some markdown
st.markdown('Streamlit makes it easy to build **interactive** apps.')

# Add a success message
st.success('App is running!')
```

*   `import streamlit as st`: This line imports the Streamlit library, commonly aliased as `st`.
*   `st.title(...)`: Displays a large title at the top of your app.
*   `st.write(...)`: A versatile command that can display text, data, plots, and more.
*   `st.header(...)`: Displays a section header.
*   `st.markdown(...)`: Allows you to write content using Markdown syntax.
*   `st.success(...)`: Displays a colored success box.

### How to run a Streamlit app

To run your `hello_world.py` app, open your terminal or command prompt, navigate to the directory where you saved the file, and run the following command:

```bash
streamlit run hello_world.py
```

Streamlit will start a local web server and open your default web browser to display the app. You should see something like this:

![Screenshot of Hello World App](hello_world_screenshot.png)

The terminal will show the local URL (usually `http://localhost:8501`) and a network URL if you want to access it from another device on the same network.

To stop the app, press `Ctrl+C` in the terminal.

## 🛠️ Part 2: Streamlit Basics

Streamlit provides a wide range of elements to build your app's user interface and display content.

### Core components: `st.title`, `st.text`, `st.markdown`, `st.write`

We've already seen `st.title`, `st.write`, and `st.markdown`.

*   `st.title('Your App Title')`: Displays the main title.
*   `st.header('Section Header')`: Displays a header for a section.
*   `st.subheader('Sub-section Header')`: Displays a sub-header.
*   `st.text('Plain text')`: Displays fixed-width text.
*   `st.markdown('**Bold** and *italic* text')`: Displays text formatted with Markdown.
*   `st.write('Anything you want to display')`: The magic function that can handle almost anything (text, numbers, dataframes, plots, etc.).

Example:

```python
import streamlit as st

st.title('Core Components Example')
st.text('This is plain text.')
st.markdown('This is **markdown** text.')
st.write('This is displayed using st.write.')
```

### Input widgets: `st.button`, `st.slider`, `st.text_input`, `st.selectbox`, etc.

Widgets allow users to interact with your app. When a widget's value changes, Streamlit re-runs the script.

*   `st.button('Click me')`: Displays a button. Returns `True` when clicked, `False` otherwise.
*   `st.checkbox('Check me')`: Displays a checkbox. Returns `True` if checked, `False` otherwise.
*   `st.radio('Choose one', ['Option A', 'Option B'])`: Displays radio buttons. Returns the selected option.
*   `st.selectbox('Select an option', ['Option 1', 'Option 2'])`: Displays a dropdown menu. Returns the selected option.
*   `st.multiselect('Select multiple', ['A', 'B', 'C'])`: Displays a multi-select box. Returns a list of selected options.
*   `st.slider('Select a value', 0, 100, 50)`: Displays a slider. Returns the current value. Parameters are (label, min_value, max_value, default_value).
*   `st.text_input('Enter your name')`: Displays a text input field. Returns the entered string.
*   `st.number_input('Enter a number', 0, 100)`: Displays a number input field. Returns the entered number.
*   `st.date_input('Pick a date')`: Displays a date picker. Returns a date object.
*   `st.time_input('Pick a time')`: Displays a time picker. Returns a time object.

Example:

```python
import streamlit as st

st.title('Widget Example')

# Button
if st.button('Say Hello'):
    st.write('Hello!')

# Checkbox
checkbox_state = st.checkbox('Show/Hide text')
if checkbox_state:
    st.write('Checkbox is checked!')

# Radio buttons
radio_option = st.radio('Choose a color', ['Red', 'Green', 'Blue'])
st.write(f'You chose: {radio_option}')

# Selectbox
selectbox_option = st.selectbox('Select a fruit', ['Apple', 'Banana', 'Cherry'])
st.write(f'You selected: {selectbox_option}')

# Slider
age = st.slider('Select your age', 0, 100, 25)
st.write(f'Your age is: {age}')

# Text input
name = st.text_input('Enter your name')
if name: # Check if name is not empty
    st.write(f'Hello, {name}!')
```

![Screenshot of Widgets Example](widgets_screenshot.png)

### Displaying data: `st.dataframe`, `st.table`, `st.json`

Streamlit makes it easy to display data structures.

*   `st.dataframe(df)`: Displays an interactive table using Pandas DataFrames. Allows sorting, resizing columns, etc.
*   `st.table(df)`: Displays a static table.
*   `st.json(data)`: Displays a JSON object in an interactive format.

Example:

```python
import streamlit as st
import pandas as pd
import json

st.title('Data Display Example')

# Create a sample DataFrame
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
df = pd.DataFrame(data)

st.subheader('st.dataframe')
st.dataframe(df) # Interactive table

st.subheader('st.table')
st.table(df) # Static table

# Create a sample JSON object
json_data = {
    "name": "Streamlit",
    "version": "1.0",
    "features": ["widgets", "data display", "charts"]
}

st.subheader('st.json')
st.json(json_data)
```

![Screenshot of Data Display Example](data_display_screenshot.png)

### Adding charts with `st.line_chart`, `st.bar_chart`, and `st.pyplot`

Visualizing data is crucial, and Streamlit integrates well with popular plotting libraries.

*   `st.line_chart(data)`: Displays a line chart.
*   `st.bar_chart(data)`: Displays a bar chart.
*   `st.area_chart(data)`: Displays an area chart.
*   `st.pyplot(fig)`: Displays a Matplotlib figure. You can use this for more complex plots from Matplotlib, Seaborn, etc.

Example:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Chart Example')

# Generate some random data
chart_data = pd.DataFrame(
    np.random.randn(20, 3), # 20 rows, 3 columns
    columns=['a', 'b', 'c']
)

st.subheader('st.line_chart')
st.line_chart(chart_data)

st.subheader('st.bar_chart')
st.bar_chart(chart_data)

st.subheader('st.area_chart')
st.area_chart(chart_data)

# Example using Matplotlib
st.subheader('st.pyplot (Matplotlib)')

fig, ax = plt.subplots() # Create a figure and an axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3]) # Plot some data on the axes.
ax.set_title('Simple Matplotlib Plot') # Add a title
ax.set_xlabel('X-axis') # Add x-label
ax.set_ylabel('Y-axis') # Add y-label

st.pyplot(fig) # Display the Matplotlib figure
```

![Screenshot of Chart Example](chart_screenshot.png)

### Using layout: columns, sidebar

You can organize your app's layout using columns and a sidebar.

*   `st.sidebar`: Everything added to `st.sidebar` will appear in a sidebar on the left.
*   `st.columns(n)`: Creates `n` columns. You can then add elements to each column.

Example:

```python
import streamlit as st

st.title('Layout Example')

# Add content to the sidebar
st.sidebar.header('Sidebar Content')
st.sidebar.text_input('Enter something in sidebar')

# Create columns
col1, col2, col3 = st.columns(3)

# Add content to columns
with col1:
    st.header('Column 1')
    st.write('This is in the first column.')

with col2:
    st.header('Column 2')
    st.write('This is in the second column.')

with col3:
    st.header('Column 3')
    st.write('This is in the third column.')

# Content outside columns and sidebar
st.header('Main Content Area')
st.write('This is in the main content area below the columns.')
```

![Screenshot of Layout Example](layout_screenshot.png)

## ✅ Part 3: Projects

Let's build a few simple projects to practice what we've learned.

### 🟢 Simple Project 1: BMI Calculator

This app will take height and weight as input and calculate the Body Mass Index (BMI).

Create a new file named `bmi_calculator.py` and add the following code:

```python
import streamlit as st

st.title('BMI Calculator')

# Input for height (in cm)
height = st.number_input('Enter your height in cm', min_value=1, format="%f") # Use %f for float input

# Input for weight (in kg)
weight = st.number_input('Enter your weight in kg', min_value=1, format="%f") # Use %f for float input

# Button to calculate BMI
calculate_button = st.button('Calculate BMI')

# Check if the button is clicked and inputs are valid
if calculate_button:
    if height > 0 and weight > 0:
        # Convert height from cm to meters
        height_in_meters = height / 100

        # Calculate BMI
        bmi = weight / (height_in_meters ** 2)

        # Display the calculated BMI
        st.subheader(f'Your BMI is: {bmi:.2f}') # Display BMI rounded to 2 decimal places

        # Determine BMI category
        if bmi < 18.5:
            st.warning('Category: Underweight') # Display warning for underweight
        elif 18.5 <= bmi < 25:
            st.success('Category: Normal weight') # Display success for normal weight
        elif 25 <= bmi < 30:
            st.warning('Category: Overweight') # Display warning for overweight
        else:
            st.error('Category: Obesity') # Display error for obesity
    else:
        st.error('Please enter valid height and weight.') # Error message for invalid input

```

Run this app using `streamlit run bmi_calculator.py`.

*   We use `st.number_input` for height and weight, ensuring they are numbers. `min_value=1` prevents division by zero or negative values. `format="%f"` ensures float input.
*   `st.button('Calculate BMI')` creates the button. The code inside the `if calculate_button:` block only runs when the button is clicked.
*   We perform the BMI calculation and then use conditional statements (`if/elif/else`) to determine the category.
*   `st.subheader`, `st.warning`, `st.success`, and `st.error` are used to display the results and category with different styles.

![Screenshot of BMI Calculator](bmi_calculator_screenshot.png)

### 🟢 Simple Project 2: Random Quote Generator

This app will display a random motivational quote when a button is clicked.

Create a new file named `quote_generator.py` and add the following code:

```python
import streamlit as st
import random # Import the random library

st.title('Random Quote Generator')

# List of motivational quotes
quotes = [
    "The best way to predict the future is to create it. - Peter Drucker",
    "Believe you can and you're halfway there. - Theodore Roosevelt",
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Strive not to be a success, but rather to be of value. - Albert Einstein",
    "Your time is limited, don't waste it living someone else's life. - Steve Jobs"
]

# Button to generate a quote
generate_button = st.button('Generate New Quote')

# Check if the button is clicked
if generate_button:
    # Select a random quote from the list
    random_quote = random.choice(quotes)

    # Display the random quote
    st.write(f"_{random_quote}_") # Display the quote in italics

# Optional: Display a default quote when the app first loads
# st.write("Click the button to get a motivational quote!")

```

Run this app using `streamlit run quote_generator.py`.

*   We import the `random` library to pick a random item from a list.
*   We define a list called `quotes` containing strings.
*   `st.button('Generate New Quote')` creates the button.
*   When the button is clicked, `random.choice(quotes)` selects a random quote.
*   `st.write()` displays the selected quote.

![Screenshot of Quote Generator](quote_generator_screenshot.png)

### 🔵 Intermediate Project 1: To-Do List Web App

This app will allow users to add, complete, and delete tasks. We will use `st.session_state` to maintain the state of the to-do list across app re-runs.

Create a new file named `todo_app.py` and add the following code:

```python
import streamlit as st

st.title('Simple To-Do List App')

# Initialize session state for tasks if it doesn't exist
# st.session_state allows us to store and persist variables across script re-runs
if 'tasks' not in st.session_state:
    st.session_state.tasks = [] # Initialize tasks as an empty list

# Input field for new task
new_task = st.text_input('Add a new task:')

# Button to add task
if st.button('Add Task') and new_task: # Check if button is clicked AND input is not empty
    # Add the new task to the tasks list in session state
    # Each task is a dictionary with 'task' and 'completed' status
    st.session_state.tasks.append({'task': new_task, 'completed': False})
    # Clear the input field after adding (optional, requires a bit more advanced handling or re-running)
    # For simplicity here, the input field will clear on re-run after button click

st.subheader('Tasks:')

# Display tasks using columns for layout
# We iterate through the tasks list with their index
for i, task_item in enumerate(st.session_state.tasks):
    # Create two columns for each task: one for checkbox/text, one for delete button
    col1, col2 = st.columns([0.8, 0.2]) # Adjust column width ratio

    with col1:
        # Create a checkbox for task completion status
        # The key is important to uniquely identify each checkbox across re-runs
        # The value of the checkbox is stored directly in the task_item dictionary
        checkbox_state = st.checkbox(
            task_item['task'], # Label for the checkbox is the task description
            value=task_item['completed'], # Initial state of the checkbox
            key=f"checkbox_{i}" # Unique key for this checkbox
        )
        # Update the completed status in session state when checkbox state changes
        st.session_state.tasks[i]['completed'] = checkbox_state

    with col2:
        # Create a delete button for each task
        # The key is important to uniquely identify each button
        delete_button = st.button(
            'Delete',
            key=f"delete_{i}" # Unique key for this button
        )
        # If the delete button is clicked, remove the task from the list
        if delete_button:
            # We need to remove the task by its index
            # To avoid issues with changing indices during iteration,
            # we can mark for deletion and clean up later, or re-run.
            # Streamlit's re-run handles this simply: the list is rebuilt without the deleted item.
            del st.session_state.tasks[i]
            # Streamlit will re-run the script automatically after deletion

# Note: Deleting items while iterating can sometimes be tricky in Python.
# Streamlit's re-run model simplifies this; when `del st.session_state.tasks[i]` is called,
# the script re-runs from the top, and the loop rebuilds the display based on the modified list.

# Optional: Display session state for debugging
# st.write(st.session_state)
```

Run this app using `streamlit run todo_app.py`.

*   `st.session_state`: This is a dictionary-like object that persists across script re-runs. We use it to store our list of tasks. We initialize it if it doesn't exist.
*   Each task is stored as a dictionary `{'task': 'description', 'completed': False}`.
*   When the "Add Task" button is clicked and the input is not empty, we append a new task dictionary to `st.session_state.tasks`.
*   We loop through the `st.session_state.tasks` list to display each task.
*   `st.columns([0.8, 0.2])` creates two columns with a width ratio of 80/20.
*   Inside the first column, `st.checkbox` displays the task description and its completion status. The `key` parameter is crucial for Streamlit to correctly manage the state of each individual checkbox when the script re-runs.
*   Inside the second column, `st.button('Delete', key=f"delete_{i}")` creates a delete button for each task. Again, a unique `key` is needed.
*   When a delete button is clicked, `del st.session_state.tasks[i]` removes the task from the list. Streamlit automatically re-runs, updating the display.

![Screenshot of To-Do App](todo_app_screenshot.png)

### 🔵 Intermediate Project 2: Weather App (using OpenWeatherMap API)

This app will fetch and display weather information for a given city using a public API.

**Prerequisites:**

*   You need an API key from OpenWeatherMap. You can get one for free from their website ([https://openweathermap.org/api](https://openweathermap.org/api)).
*   You need the `requests` library installed: `pip install requests`.

Create a new file named `weather_app.py` and add the following code:

```python
import streamlit as st
import requests # Import the requests library to make HTTP requests
import json # Import json to parse the API response

st.title('Simple Weather App')

# --- Configuration ---
# Replace with your actual OpenWeatherMap API key
# It's better practice to store API keys securely (e.g., environment variables)
# but for this tutorial, we'll put it directly here.
API_KEY = 'YOUR_OPENWEATHERMAP_API_KEY' # <-- Replace with your key!
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather?' # Base URL for the weather API

# --- Input ---
city_name = st.text_input('Enter city name:')

# --- Fetch and Display Weather ---
# Check if a city name has been entered
if city_name:
    # Construct the full API URL
    # We request temperature in Celsius (&units=metric)
    complete_url = BASE_URL + 'appid=' + API_KEY + '&q=' + city_name + '&units=metric'

    # Make the GET request to the API
    response = requests.get(complete_url)

    # Parse the JSON response
    data = response.json()

    # Check the API response status code
    # 'cod' 200 means success
    if data['cod'] == 200:
        # Extract relevant weather information from the JSON data
        main_data = data['main'] # Contains temperature, pressure, humidity
        weather_data = data['weather'][0] # Contains weather condition, description, icon

        # Get temperature, pressure, humidity
        temperature = main_data['temp']
        pressure = main_data['pressure']
        humidity = main_data['humidity']

        # Get weather description and icon code
        weather_description = weather_data['description']
        weather_icon_code = weather_data['icon'] # e.g., '01d' for clear sky day

        # Construct the URL for the weather icon
        icon_url = f"http://openweathermap.org/img/wn/{weather_icon_code}@2x.png"

        # Display the weather information
        st.subheader(f"Weather in {city_name.capitalize()}:") # Capitalize city name for display
        st.write(f"Temperature: {temperature}°C")
        st.write(f"Humidity: {humidity}%")
        st.write(f"Pressure: {pressure} hPa")
        st.write(f"Condition: {weather_description.capitalize()}") # Capitalize description

        # Display the weather icon
        st.image(icon_url, caption=weather_description.capitalize())

    else:
        # Handle errors (e.g., city not found)
        st.error(f"City '{city_name}' not found. Please check the spelling.")
        # Optional: Display the full error message from the API
        # st.json(data)

```

**Remember to replace `'YOUR_OPENWEATHERMAP_API_KEY'` with your actual API key!**

Run this app using `streamlit run weather_app.py`.

*   We import `requests` to fetch data from the web and `json` to parse the response.
*   We define the `API_KEY` and `BASE_URL`. **Crucially, replace the placeholder API key with your own.**
*   `st.text_input` gets the city name from the user.
*   When a city name is entered, we construct the full API request URL.
*   `requests.get(complete_url)` sends the request to the OpenWeatherMap API.
*   `response.json()` parses the JSON response into a Python dictionary.
*   We check `data['cod'] == 200` to see if the request was successful.
*   If successful, we extract the temperature, humidity, pressure, description, and icon code from the `data` dictionary.
*   We construct a URL for the weather icon image provided by OpenWeatherMap.
*   `st.subheader`, `st.write`, and `st.image` are used to display the results.
*   If `data['cod']` is not 200, it means there was an error (like an invalid city name), and we display an error message using `st.error`.

![Screenshot of Weather App](weather_app_screenshot.png)

## 📄 Part 4: Deployment

Once your Streamlit app is ready, you'll likely want to share it with others. Streamlit provides a free hosting service called Streamlit Community Cloud.

Here are the general steps:

### Creating `requirements.txt`

Streamlit Community Cloud needs to know which Python packages your app depends on. You create a file named `requirements.txt` that lists these dependencies.

Navigate to your project directory in the terminal and run:

```bash
pip freeze > requirements.txt
```

This command lists all installed packages in your current Python environment and saves them to `requirements.txt`.

**Important:** If you are working in a virtual environment that only contains the packages needed for your app (like `streamlit`, `pandas`, `requests`, `numpy`, `matplotlib`), this command will generate a clean `requirements.txt`. If you run this in your global environment, it might include many unnecessary packages. Using virtual environments is highly recommended.

Your `requirements.txt` for the projects above might look something like this (versions might differ):

```
streamlit>=1.0.0
pandas>=1.0.0
numpy>=1.0.0
matplotlib>=3.0.0
requests>=2.0.0
```

### Setting up the GitHub repository

Streamlit Community Cloud deploys directly from GitHub repositories.

1.  Create a new repository on GitHub ([https://github.com/new](https://github.com/new)).
2.  Initialize a Git repository in your local project folder:
    ```bash
    git init
    ```
3.  Add your files (your `.py` app file and `requirements.txt`):
    ```bash
    git add .
    ```
4.  Commit your changes:
    ```bash
    git commit -m "Initial commit: Add Streamlit app and requirements"
    ```
5.  Link your local repository to the GitHub repository (replace `YOUR_GITHUB_USERNAME` and `YOUR_REPO_NAME`):
    ```bash
    git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
    ```
6.  Push your code to GitHub:
    ```bash
    git push -u origin main # Or 'master' depending on your default branch name
    ```

Make sure your main app file (e.g., `hello_world.py`, `bmi_calculator.py`) is at the root of the repository or in a clearly defined subdirectory.

### Pushing code and linking to Streamlit

1.  Go to Streamlit Community Cloud ([https://share.streamlit.io/](https://share.streamlit.io/)).
2.  Log in or sign up.
3.  Click "Deploy an app".
4.  Select "From existing repo".
5.  Connect your GitHub account if you haven't already.
6.  Choose your repository from the dropdown list.
7.  Specify the branch (usually `main` or `master`).
8.  Specify the main file path (e.g., `bmi_calculator.py`).
9.  Click "Deploy!".

Streamlit will read your `requirements.txt`, install the dependencies, and build your app. Once deployed, you'll get a public URL to share your app!

![Screenshot of Streamlit Cloud Deployment](streamlit_cloud_deploy_screenshot.png)

## Conclusion

Congratulations! You've learned the basics of building and deploying Streamlit web apps. We covered core components, widgets, data display, charts, layout, and built a few simple projects.

Streamlit has many more features to explore, including caching, theming, connecting to databases, and more. The best way to learn is to start building your own projects!

Happy Streamlit-ing!
