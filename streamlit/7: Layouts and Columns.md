## 🔹 **Step 7: Layouts and Columns**

In this step, you'll learn how to **arrange** your app neatly using:

* **Columns** (side-by-side)
* **Expander** (collapsible content)
* **Sidebar** (for filters, options)
* **Containers** (grouping widgets or sections)

---

### ✅ 1. **Columns** – Side-by-side layout

```python
import streamlit as st

col1, col2 = st.columns(2)

with col1:
    st.write("👈 This is Column 1")
    st.button("Click me!")

with col2:
    st.write("👉 This is Column 2")
    st.slider("Pick a number", 0, 100)
```

> Use it to display cards, filters, forms, or data next to each other.

---

### ✅ 2. **Expander** – Hide/Show on click

```python
with st.expander("Click to see more"):
    st.write("Here is some hidden content.")
    st.image("https://picsum.photos/200")
```

> Good for showing **extra info** or instructions.

---

### ✅ 3. **Sidebar** – Clean up the main page

Use `st.sidebar` to move widgets to a sidebar.

```python
st.sidebar.title("Options")
selected = st.sidebar.selectbox("Choose one:", ["Home", "About", "Contact"])
st.write("You selected:", selected)
```

> Helpful for menus, filters, controls, and forms.

---

### ✅ 4. **Container** – Group items together

```python
container = st.container()
container.write("This is inside a container")
container.line_chart([1, 5, 2, 6])
```

> Use when you want to **group or update** elements later.

---

### ✅ 5. **Empty** – Placeholder for future content

```python
placeholder = st.empty()

import time
for i in range(5):
    placeholder.text(f"Loading... {i}")
    time.sleep(1)

placeholder.text("Done!")
```

> Useful when updating UI elements dynamically (like showing progress).

---

### 🧪 Combine Them Together

```python
import streamlit as st

st.title("Layout Example")

# Sidebar
st.sidebar.title("Sidebar Menu")
name = st.sidebar.text_input("Your name", "Yogesh")
st.sidebar.write("Hello,", name)

# Columns
col1, col2 = st.columns(2)
with col1:
    st.header("Left Panel")
    st.button("Button A")

with col2:
    st.header("Right Panel")
    st.checkbox("Check me")

# Expander
with st.expander("Show more"):
    st.write("Here’s some additional info!")

# Container
with st.container():
    st.subheader("Grouped Section")
    st.write("Everything here is inside a container.")
```

---

### 🧠 Summary

| Layout Tool      | Purpose                                 |
| ---------------- | --------------------------------------- |
| `st.columns()`   | Side-by-side layout                     |
| `st.expander()`  | Collapsible sections                    |
| `st.sidebar`     | Cleaner UI for filters & menus          |
| `st.container()` | Group related elements together         |
| `st.empty()`     | Placeholder for dynamic content updates |

---

✅ That’s it for layouts!

