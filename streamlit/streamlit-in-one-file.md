# Streamlit Tutorial: Build Interactive Data Apps in Python

```markdown
# Streamlit Tutorial: Build Interactive Data Apps in Python

## 1. Introduction

### What is Streamlit?
Streamlit is an **open-source Python framework** that lets you create beautiful, interactive web applications for data science and machine learning with minimal code. It turns Python scripts into shareable web apps in minutes.

### Key Features and Benefits
- ⚡ **Rapid development**: Create apps with simple Python scripts
- 🎨 **No front-end knowledge required**: Focus on Python logic instead of HTML/JavaScript
- 📊 **Built-in widgets**: Sliders, buttons, file uploaders, and more
- 🔄 **Automatic reactivity**: Apps automatically update when code changes
- � **Seamless integration**: Works with Pandas, Matplotlib, Plotly, PyTorch, TensorFlow, etc.

### Installation
Install Streamlit using pip:
```bash
pip install streamlit
```
Verify installation:
```bash
streamlit hello
```
This command will open a demo app in your browser.

### How Streamlit Works (High-Level Overview)
1. You write a Python script using Streamlit commands
2. Streamlit runs your script from top to bottom
3. When a user interacts with a widget, Streamlit re-runs the script
4. Output is rendered in the browser

---

## 2. Getting Started

### Your First Streamlit App
Create a file `hello_streamlit.py`:
```python
import streamlit as st

st.title("My First Streamlit App")
st.write("Hello, Streamlit!")

name = st.text_input("What's your name?")
if name:
    st.success(f"Hello, {name}! 👋")
```

Run the app from terminal:
```bash
streamlit run hello_streamlit.py
```

### Basic Components
- **Text elements**:
  ```python
  st.title("Main Title")
  st.header("Section Header")
  st.subheader("Subsection")
  st.text("Fixed-width text")
  st.markdown("**Markdown** support!")
  ```

- **Input widgets**:
  ```python
  # Text input
  user_input = st.text_input("Enter text")
  
  # Button
  if st.button("Click me"):
      st.write("Button clicked!")
  
  # Slider
  age = st.slider("Your age", 0, 100, 25)
  
  # Checkbox
  agree = st.checkbox("I agree")
  
  # Selectbox
  option = st.selectbox("Choose one", ["A", "B", "C"])
  ```

---

## 3. Project 1: BMI Calculator

Create `bmi_calculator.py`:
```python
import streamlit as st

st.title("BMI Calculator")

with st.form("bmi_form"):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
    with col2:
        height = st.number_input("Height (m)", min_value=0.5, value=1.75, step=0.01)
    
    submitted = st.form_submit_button("Calculate BMI")
    
    if submitted:
        bmi = weight / (height ** 2)
        st.subheader(f"Your BMI: {bmi:.1f}")
        
        if bmi < 18.5:
            st.warning("Underweight")
            st.image("https://via.placeholder.com/150/FFFF00/000000?text=Underweight")
        elif 18.5 <= bmi < 25:
            st.success("Healthy")
            st.image("https://via.placeholder.com/150/00FF00/000000?text=Healthy")
        elif 25 <= bmi < 30:
            st.warning("Overweight")
            st.image("https://via.placeholder.com/150/FFA500/000000?text=Overweight")
        else:
            st.error("Obese")
            st.image("https://via.placeholder.com/150/FF0000/FFFFFF?text=Obese")
```

**Features**:
- Form layout with columns
- BMI calculation
- Conditional results with images
- Styled messages (success, warning, error)

---

## 4. Project 2: Currency Converter

Create `currency_converter.py`:
```python
import streamlit as st
import requests

st.title("💰 Currency Converter")

# Get currency data (using free API)
@st.cache_data
def get_currencies():
    response = requests.get("https://open.er-api.com/v6/latest/USD")
    return response.json()["rates"]

currencies = get_currencies()

with st.form("conversion_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input("Amount", min_value=0.01, value=100.0)
    with col2:
        from_currency = st.selectbox("From", list(currencies.keys()))
    with col3:
        to_currency = st.selectbox("To", list(currencies.keys()))
    
    submitted = st.form_submit_button("Convert")
    
    if submitted:
        # Convert via USD as base
        usd_amount = amount / currencies[from_currency]
        result = usd_amount * currencies[to_currency]
        st.success(f"**{amount} {from_currency} = {result:.2f} {to_currency}**")
        st.balloons()
```

**Features**:
- Real-time currency rates via API
- Caching with `@st.cache_data`
- Multi-column layout
- Visual feedback (success message + balloons)

---

## 5. Intermediate Concepts

### Layouts and Columns
Organize content into tabs, columns, and expanders:
```python
# Columns
col1, col2 = st.columns(2)
with col1:
    st.header("Column 1")
    st.line_chart([1, 2, 3, 4])
with col2:
    st.header("Column 2")
    st.area_chart([4, 3, 2, 1])

# Tabs
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("Content for Tab 1")
with tab2:
    st.write("Content for Tab 2")

# Expander
with st.expander("See explanation"):
    st.write("Hidden details appear here!")
```

### Caching
Speed up apps with caching:
```python
@st.cache_data  # Cache data loading
def load_data(url):
    return pd.read_csv(url)

data = load_data("https://data.example.com/large_dataset.csv")
```

### File Upload and Download
```python
# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df)

# Download button
st.download_button(
    label="Download data as CSV",
    data=df.to_csv().encode('utf-8'),
    file_name='data.csv',
    mime='text/csv'
)
```

### Data Visualization
Integrate popular plotting libraries:
```python
import matplotlib.pyplot as plt
import plotly.express as px

# Matplotlib
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 1])
st.pyplot(fig)

# Plotly
fig = px.scatter(df, x='x_col', y='y_col', color='category')
st.plotly_chart(fig)

# Streamlit built-in
st.line_chart(df[['col1', 'col2']])
```

---

## 6. Project 3: Data Explorer

Create `data_explorer.py`:
```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Data Explorer")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data")
    st.dataframe(df.head())
    
    st.subheader("Data Summary")
    st.write(df.describe())
    
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        numeric_cols = df.select_dtypes(include=['number']).columns
        selected_col = st.selectbox("Select column", numeric_cols)
    with col2:
        min_val = float(df[selected_col].min())
        max_val = float(df[selected_col].max())
        value_range = st.slider("Filter range", min_val, max_val, (min_val, max_val))
    
    filtered_df = df[(df[selected_col] >= value_range[0]) & 
                     (df[selected_col] <= value_range[1])]
    
    st.subheader("Filtered Data")
    st.write(f"Rows: {len(filtered_df)}")
    st.dataframe(filtered_df)
    
    st.subheader("Visualization")
    plot_type = st.selectbox("Choose plot type", ["Histogram", "Scatter Plot"])
    
    if plot_type == "Histogram":
        fig = px.histogram(filtered_df, x=selected_col)
        st.plotly_chart(fig)
    else:
        col_x = st.selectbox("X-axis", numeric_cols)
        col_y = st.selectbox("Y-axis", numeric_cols)
        fig = px.scatter(filtered_df, x=col_x, y=col_y)
        st.plotly_chart(fig)
```

**Features**:
- CSV file upload
- Data preview and summary statistics
- Interactive filtering with slider
- Dynamic Plotly visualizations

---

## 7. Project 4: Sentiment Analyzer

Create `sentiment_analyzer.py`:
```python
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt

st.title("🧠 Sentiment Analyzer")

text = st.text_area("Enter your text:", "Streamlit makes building data apps easy and fun!")

if st.button("Analyze"):
    with st.spinner("Analyzing sentiment..."):
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        col1.metric("Polarity", f"{sentiment.polarity:.2f}")
        col2.metric("Subjectivity", f"{sentiment.subjectivity:.2f}")
        
        # Visualize sentiment
        fig, ax = plt.subplots()
        ax.barh(["Polarity", "Subjectivity"], [sentiment.polarity, sentiment.subjectivity], color=['blue', 'green'])
        ax.set_xlim(-1, 1)
        ax.set_title("Sentiment Analysis")
        st.pyplot(fig)
        
        # Interpretation
        if sentiment.polarity > 0.5:
            st.success("😊 Strongly Positive")
        elif sentiment.polarity > 0:
            st.success("🙂 Positive")
        elif sentiment.polarity == 0:
            st.info("😐 Neutral")
        elif sentiment.polarity > -0.5:
            st.warning("🙁 Negative")
        else:
            st.error("😠 Strongly Negative")

st.caption("Note: Using TextBlob for sentiment analysis. Install with `pip install textblob`")
```

**Features**:
- Real-time sentiment analysis
- Dynamic metrics display
- Horizontal bar chart visualization
- Emoji-based sentiment interpretation
- Loading spinner during processing

---

## 8. Deployment

### Streamlit Cloud Deployment (Recommended)
1. Create GitHub repository with your Streamlit app
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" → Select repo and branch
4. Specify app file path (e.g., `app.py`)
5. Click "Deploy"

### Alternative Deployment Options
- **Hugging Face Spaces**:
  1. Create account at [huggingface.co](https://huggingface.co)
  2. Create new Space → Select Streamlit template
  3. Add your `app.py` and `requirements.txt`
  
- **Heroku**:
  1. Create `requirements.txt` and `Procfile`
  2. Install Heroku CLI and login
  3. Run:
     ```bash
     heroku create
     git push heroku main
     ```

### Sample requirements.txt
```
streamlit==1.25.0
pandas==2.0.3
plotly==5.15.0
textblob==0.17.1
requests==2.31.0
```

---

## 9. Conclusion

### Next Steps
- Explore [Streamlit Component Gallery](https://streamlit.io/gallery)
- Try advanced features: [State management](https://docs.streamlit.io/library/api-reference/session-state), [theming](https://docs.streamlit.io/library/advanced-features/theming), [multipage apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- Build your own components with React

### Project Ideas
- 🗺️ Real-time COVID tracker with maps
- 📈 Stock market dashboard
- 🖼️ Image classifier with TensorFlow/PyTorch
- 💬 Real-time chat application
- 🎮 Game with Streamlit components

### Resources
- [Official Documentation](https://docs.streamlit.io)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [Awesome Streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)

**Happy Streamlit building!** 🚀
```

This comprehensive tutorial includes:
- Beginner-friendly explanations
- Complete runnable code for 4 projects
- Intermediate concepts with examples
- Deployment instructions
- Visual elements and formatting for clarity

To use this tutorial:
1. Save as `streamlit_tutorial.md`
2. Run each project with `streamlit run filename.py`
3. Experiment with modifying the code examples

The tutorial progresses from basic concepts to intermediate projects, giving learners a practical foundation for building their own Streamlit applications.
