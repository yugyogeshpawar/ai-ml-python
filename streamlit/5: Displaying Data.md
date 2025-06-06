## 🔹 **Step 5: Displaying Data**

Streamlit makes it **super easy** to display tables, DataFrames, and styled data. Great for AI/ML and data apps!

---

### 🧠 Why Display Data?

You often want to:

* Show model results
* Visualize CSV/Excel data
* Present summaries

Streamlit can handle `pandas`, `NumPy`, lists, dictionaries, and more.

---

### 🐼 Let's Start with `pandas`

First, install pandas if you haven't already:

```bash
pip install pandas
```

Then import it:

```python
import pandas as pd
```

---

### ✅ 1. **st.table()** – Static Table

Use this when you want a **clean, fixed table** (not interactive):

```python
import streamlit as st
import pandas as pd

data = pd.DataFrame({
    'Name': ['Yogesh', 'Yuvraj', 'Anuj'],
    'Score': [85, 92, 78]
})

st.table(data)
```

---

### ✅ 2. **st.dataframe()** – Interactive Table

Use this when you want to:

* Scroll
* Sort
* Resize

```python
st.dataframe(data)
```

> 🔄 You can update it live inside your app.

---

### ✅ 3. **st.write()** – Smart Display

`st.write()` can show tables too:

```python
st.write(data)
```

---

### ✅ 4. **Highlighting Data**

You can style your DataFrame with colors:

```python
styled_df = data.style.highlight_max(axis=0)
st.dataframe(styled_df)
```

This will highlight the **maximum value in each column**.

---

### ✅ 5. **Displaying CSV Data**

Want to load and show a CSV file?

```python
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here's your data:")
    st.dataframe(df)
```

---

### ✅ 6. **Showing Dictionary / List**

```python
my_dict = {"Python": 95, "Java": 90, "C++": 88}
st.write(my_dict)

my_list = ["Apple", "Banana", "Mango"]
st.write(my_list)
```

---

### 🧪 Full Example

```python
import streamlit as st
import pandas as pd

st.title("Displaying Data in Streamlit")

data = pd.DataFrame({
    "Name": ["Yogesh", "Nidhi", "Tushar"],
    "Age": [27, 24, 30],
    "Score": [88, 92, 85]
})

st.subheader("Static Table")
st.table(data)

st.subheader("Interactive DataFrame")
st.dataframe(data)

st.subheader("Styled Data")
st.dataframe(data.style.highlight_max(axis=0))
```

---

### 🧠 Summary

| Function                               | Purpose                          |
| -------------------------------------- | -------------------------------- |
| `st.table()`                           | Static, clean table              |
| `st.dataframe()`                       | Interactive, scrollable table    |
| `st.write()`                           | Smart display of almost anything |
| `pandas.read_csv()` + `st.dataframe()` | Display uploaded CSV             |

