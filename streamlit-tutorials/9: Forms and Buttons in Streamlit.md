## 🔹 **Step 9: Forms and Buttons in Streamlit**

Streamlit **reruns the entire script** from top to bottom on every interaction. But when you want to control **when** inputs get submitted (like a form), you can use:

* ✅ `st.form` — for grouped form inputs
* ✅ `st.button` — for simple one-click actions
* ✅ `st.form_submit_button` — for full form submission

---

## 🔸 1. **Simple Button**

```python
import streamlit as st

if st.button("Say Hello"):
    st.write("Hello, Yogesh 👋")
else:
    st.write("Click the button above.")
```

---

## 🔸 2. **Form with Multiple Inputs**

```python
with st.form("my_form"):
    name = st.text_input("Enter your name")
    age = st.number_input("Your age", min_value=1, max_value=100)
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.success(f"Welcome, {name}! Age: {int(age)}")
```

> All inputs are grouped, and **nothing happens** until the Submit button is pressed.

---

## 🔸 3. **Login Form with Session State**

```python
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Login")

        if login:
            if username == "yogesh" and password == "1234":
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
            else:
                st.error("❌ Invalid credentials")
else:
    st.success("🎉 You are logged in!")
    st.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))
```

---

## 🔸 4. **Feedback Form Example**

```python
with st.form("feedback_form"):
    rating = st.slider("Rate your experience (1-10)", 1, 10)
    comments = st.text_area("Any comments?")
    send = st.form_submit_button("Send Feedback")

    if send:
        st.success(f"Thank you for rating us {rating}/10! 😊")
        if comments:
            st.info(f"You said: {comments}")
```

---

## 🧠 When to Use Forms

| Use Case           | Recommended Tool    |
| ------------------ | ------------------- |
| One-click actions  | `st.button()`       |
| Login/Signup       | `st.form()` + state |
| Surveys & Feedback | `st.form()`         |
| Controlled inputs  | `st.form()`         |

---

### ✅ Combine Forms with Session State

To **remember** form inputs, login states, or other interactions, use `st.session_state`.

---

### 🔄 Rerun Tip

> Forms prevent Streamlit from rerunning the app **until you click submit**, which is helpful for performance and cleaner UX.


