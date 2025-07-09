

## 🔹 **Step 2: Your First Streamlit App**

This step will help you write your **first real app** using Streamlit.

---

### 🖥️ Step 1: Create a Python File

Open your favorite code editor (like VS Code, PyCharm, or even Notepad) and create a file called:

```
first_app.py
```

---

### 🧑‍💻 Step 2: Write This Simple Code

Copy and paste this into your `first_app.py` file:

```python
import streamlit as st

st.title("Welcome to My First Streamlit App 👋")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is some basic text.")
st.write("Streamlit makes it easy to build apps with Python.")
```

Let’s break this down:

* `st.title()` – Shows a big main title
* `st.header()` – Smaller than title, good for section headings
* `st.subheader()` – Even smaller than header
* `st.text()` – Shows plain text
* `st.write()` – Smart command that can show text, numbers, tables, charts, etc.

---

### ▶️ Step 3: Run the App

Now open your terminal or command prompt, go to the folder where your file is saved, and type:

```bash
streamlit run first_app.py
```

Streamlit will start a local server and open the app in your **browser**. 🎉

---

### 🎯 Output

You will see something like this:

```
Welcome to My First Streamlit App 👋
This is a header
This is a subheader
This is some basic text.
Streamlit makes it easy to build apps with Python.
```

---

### 📝 Optional: Update and Refresh

If you change anything in the `first_app.py` file, Streamlit will **automatically detect** the change and refresh the browser.

Try changing this line:

```python
st.write("Streamlit makes it easy to build apps with Python.")
```

to:

```python
st.write("This is my updated Streamlit app 🚀")
```

Just save the file, and you’ll see the browser update. No need to restart the server.

---

### ✅ Summary

* You created your first Streamlit app with basic text.
* You learned how to run the app using `streamlit run`.
* You saw how easy it is to update and see changes.


