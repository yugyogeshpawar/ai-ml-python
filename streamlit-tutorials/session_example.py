import streamlit as st
import json
import os

DATA_FILE = "data.json"

# Load saved data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        saved_data = json.load(f)
else:
    saved_data = {"count": 0}

# Load to session state
if "count" not in st.session_state:
    st.session_state.count = saved_data["count"]

def increment():
    st.session_state.count += 1
    # Save to file
    with open(DATA_FILE, "w") as f:
        json.dump({"count": st.session_state.count}, f)

st.button("Increment", on_click=increment)
st.write("Count:", st.session_state.count)
