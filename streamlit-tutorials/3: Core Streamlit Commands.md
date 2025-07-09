
## 🔹 **Step 3: Core Streamlit Commands**

Streamlit provides simple functions to display different types of content like titles, text, markdown, code, and more. These are the **building blocks** of every Streamlit app.

---

### 🧩 1. **Title and Text Elements**

These commands show text in different styles:

```python
st.title("This is a Title")
st.header("This is a Header")
st.subheader("This is a Subheader")
st.text("This is plain text")
```

---

### ✍️ 2. **Markdown**

Markdown is a way to style text (like bold, italic, lists) using special characters.

```python
st.markdown("### This is Markdown Header")
st.markdown("**Bold Text**")
st.markdown("*Italic Text*")
st.markdown("- Bullet list item 1\n- Bullet list item 2")
st.markdown("[Click Here](https://streamlit.io)")
```

📌 *Markdown is powerful for adding styled content easily.*

---

### 📄 3. **Write Anything with `st.write()`**

This is the most flexible command. It can handle:

* Text
* Numbers
* DataFrames
* Charts
* Even custom objects

```python
st.write("This is written using st.write()")
st.write(1234)
st.write({"key": "value", "name": "Yogesh"})
```

---

### 🧮 4. **Display Code and JSON**

You can show code blocks and JSON nicely:

```python
st.code("""
def hello():
    print("Hello, Streamlit!")
""", language='python')
```

```python
st.json({
    "name": "Nidhi",
    "age": 24,
    "skills": ["Python", "ML"]
})
```

---

### ➕ Bonus: Emoji Support 😄

You can even add emojis using text or markdown:

```python
st.markdown("Hello Streamlit :wave:")
st.markdown("Python is awesome :snake:")
```

---

### ✅ Try It Out

Here’s a full example to copy and test:

```python
import streamlit as st

st.title("Core Streamlit Commands")
st.header("Text Examples")
st.subheader("Using st.text")
st.text("This is simple text")

st.header("Markdown Examples")
st.markdown("**Bold**, *Italic*, and [Link](https://streamlit.io)")

st.header("Write Examples")
st.write("This is a string")
st.write(["A", "list", "of", "items"])

st.header("Code and JSON")
st.code("for i in range(5): st.write(i)", language="python")
st.json({"framework": "Streamlit", "version": "1.0"})
```

---

### 🧠 Summary

* Use `st.title`, `st.header`, and `st.subheader` for headings.
* Use `st.markdown` for styled text (bold, italics, lists, links).
* Use `st.write` for almost anything.
* Use `st.code` and `st.json` to show code and structured data.

