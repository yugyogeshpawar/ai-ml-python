
## 🔹 **Step 8: Adding Media (Images, Videos, Audio)**

(with **Session State** support for interaction)

Streamlit allows you to **display media files** easily — like images, audio, and video.
You can also **store user actions or choices** using `st.session_state`.

---

## 🔸 Part A: Displaying Media

### ✅ 1. **Display Image**

```python
import streamlit as st
from PIL import Image

image = Image.open("your_image.jpg")  # Local file
st.image(image, caption="My Picture", use_column_width=True)

# Or from a URL:
st.image("https://picsum.photos/400", caption="Random Image")
```

---

### ✅ 2. **Play Audio**

```python
audio_file = open("example.mp3", "rb")
st.audio(audio_file.read(), format="audio/mp3")
```

> You can use `.wav`, `.mp3`, etc.

---

### ✅ 3. **Play Video**

```python
video_file = open("example.mp4", "rb")
st.video(video_file.read())
```

---

## 🔸 Part B: Using `st.session_state`

`st.session_state` is used to:

* Store data between reruns
* Track user interactions
* Maintain form input or login sessions

---

### ✅ Example 1: Counter with Session State

```python
import streamlit as st

# Initialize state variable
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment():
    st.session_state.count += 1

st.button("Increase", on_click=increment)
st.write("Count:", st.session_state.count)
```

---

### ✅ Example 2: Remember Uploaded Image

```python
import streamlit as st
from PIL import Image

st.title("Upload and View Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    st.session_state["image"] = uploaded_file  # Save in session

# Check session and show
if "image" in st.session_state:
    img = Image.open(st.session_state["image"])
    st.image(img, caption="Uploaded Image", use_column_width=True)
```

---

### ✅ Example 3: Video Upload with Preview

```python
video = st.file_uploader("Upload video", type=["mp4"])
if video:
    st.session_state["video"] = video

if "video" in st.session_state:
    st.video(st.session_state["video"])
```

---

### 🧠 Summary

| Media Type | Function           |
| ---------- | ------------------ |
| Image      | `st.image()`       |
| Audio      | `st.audio()`       |
| Video      | `st.video()`       |
| Session    | `st.session_state` |

Session state helps remember inputs even when Streamlit **reruns the script**.

