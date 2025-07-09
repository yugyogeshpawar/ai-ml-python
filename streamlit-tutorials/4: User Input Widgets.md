## 🔹 **Step 4: User Input Widgets**

Now we make your Streamlit app **interactive**!
You’ll learn how to take **input from users** using widgets like buttons, checkboxes, sliders, text fields, and more.

---

### 🧑‍💻 Why Use Widgets?

Widgets let users:

* Click buttons
* Enter text
* Select options
* Upload files
* Adjust sliders

And you can **use those inputs in your app**.

---

### 🧪 Let's See Some Examples

---

### ✅ 1. **Button**

```python
if st.button("Click Me"):
    st.write("You clicked the button!")
```

---

### ✅ 2. **Checkbox**

```python
agree = st.checkbox("I agree to the terms")
if agree:
    st.write("Thank you for agreeing!")
```

---

### ✅ 3. **Radio Buttons**

```python
option = st.radio("Choose one:", ["Option 1", "Option 2", "Option 3"])
st.write("You selected:", option)
```

---

### ✅ 4. **Selectbox**

```python
choice = st.selectbox("Pick a fruit", ["Apple", "Banana", "Mango"])
st.write("Your favorite fruit is:", choice)
```

---

### ✅ 5. **Multiselect**

```python
multi = st.multiselect("Select multiple fruits", ["Apple", "Banana", "Mango"])
st.write("You selected:", multi)
```

---

### ✅ 6. **Slider**

```python
age = st.slider("Select your age", 1, 100, 25)
st.write("Your age is:", age)
```

---

### ✅ 7. **Text Input**

```python
name = st.text_input("Enter your name", "Yogesh")
st.write("Hello,", name)
```

---

### ✅ 8. **Number Input**

```python
number = st.number_input("Enter a number", min_value=0, max_value=100, value=10)
st.write("You entered:", number)
```

---

### ✅ 9. **Date and Time Input**

```python
import datetime

d = st.date_input("Pick a date", datetime.date.today())
t = st.time_input("Pick a time", datetime.time(12, 00))
st.write("Date:", d, "Time:", t)
```

---

### ✅ 10. **File Uploader**

```python
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
```

---

### 🧠 Summary

| Widget               | Function                    |
| -------------------- | --------------------------- |
| `st.button()`        | Click actions               |
| `st.checkbox()`      | Toggle on/off               |
| `st.radio()`         | Choose one                  |
| `st.selectbox()`     | Dropdown menu               |
| `st.multiselect()`   | Choose multiple items       |
| `st.slider()`        | Pick a number with a slider |
| `st.text_input()`    | Get user text               |
| `st.file_uploader()` | Upload files                |

---

### 🎯 Try It All Together

```python
import streamlit as st
import datetime

st.title("User Input Examples")

if st.button("Click"):
    st.write("Button was clicked!")

if st.checkbox("Show message"):
    st.write("Checkbox is checked!")

fruit = st.selectbox("Pick a fruit", ["Apple", "Banana", "Mango"])
st.write("Selected:", fruit)

name = st.text_input("Your name")
st.write("Hello", name)

number = st.slider("Pick a number", 0, 50, 10)
st.write("Number is", number)

date = st.date_input("Pick a date", datetime.date.today())
st.write("Date selected:", date)
```

