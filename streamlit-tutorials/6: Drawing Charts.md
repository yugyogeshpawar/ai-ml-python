## 🔹 **Step 6: Drawing Charts**

Streamlit lets you create charts easily using:

* Built-in chart functions (`st.line_chart`, `st.bar_chart`, etc.)
* Popular libraries like **Matplotlib**, **Plotly**, and **Altair**

No need for complicated setup — just pass your data!

---

### ✅ 1. **Line Chart**

```python
import streamlit as st
import pandas as pd
import numpy as np

# Create some random data
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["Yogesh", "Anuj", "Yuvraj"]
)

st.line_chart(chart_data)
```

📈 This creates a simple **line chart** with 3 lines.

---

### ✅ 2. **Bar Chart**

```python
st.bar_chart(chart_data)
```

📊 Shows a **bar chart** using the same data.

---

### ✅ 3. **Area Chart**

```python
st.area_chart(chart_data)
```

🏞️ Like a line chart but filled underneath.

---

### ✅ 4. **Matplotlib Chart**

Install matplotlib if needed:

```bash
pip install matplotlib
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([10, 20, 30], [1, 4, 9])
st.pyplot(fig)
```

📌 Use this when you need **more control** over the chart.

---

### ✅ 5. **Plotly Chart**

Install plotly:

```bash
pip install plotly
```

```python
import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
st.plotly_chart(fig)
```

🎯 Great for **interactive charts** with tooltips, zoom, etc.

---

### ✅ 6. **Altair Chart**

Install altair:

```bash
pip install altair
```

```python
import altair as alt

df = pd.DataFrame({
    'Name': ['Yogesh', 'Anuj', 'Yuvraj'],
    'Score': [88, 92, 85]
})

chart = alt.Chart(df).mark_bar().encode(
    x='Name',
    y='Score'
)

st.altair_chart(chart, use_container_width=True)
```

🧱 Altair is **declarative**, good for layered visualizations.

---

### 🧠 Summary

| Chart Type         | Function              | Library Used       |
| ------------------ | --------------------- | ------------------ |
| Line, Bar, Area    | `st.line_chart()` etc | Streamlit built-in |
| Custom Charts      | `st.pyplot()`         | Matplotlib         |
| Interactive Charts | `st.plotly_chart()`   | Plotly             |
| Declarative        | `st.altair_chart()`   | Altair             |

---

### ✅ Try This Combined Example

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt

st.title("Chart Examples")

# Data
data = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["Yogesh", "Anuj", "Yuvraj"]
)

# Line chart
st.subheader("Line Chart")
st.line_chart(data)

# Bar chart
st.subheader("Bar Chart")
st.bar_chart(data)

# Matplotlib
st.subheader("Matplotlib Chart")
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [3, 6, 9])
st.pyplot(fig)

# Plotly
st.subheader("Plotly Chart")
fig = px.line(data, y=["Yogesh", "Anuj", "Yuvraj"])
st.plotly_chart(fig)

# Altair
st.subheader("Altair Chart")
score_df = pd.DataFrame({
    'Name': ['Yogesh', 'Anuj', 'Yuvraj'],
    'Score': [88, 92, 85]
})
chart = alt.Chart(score_df).mark_bar().encode(x='Name', y='Score')
st.altair_chart(chart, use_container_width=True)
```
