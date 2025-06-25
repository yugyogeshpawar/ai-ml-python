import streamlit as st
from PyPDF2 import PdfReader
import gemini_utils as gu
import time


st.write("Ai interviewer")

pdf_file = st.file_uploader("Upload your resume")
progress_text = "Operation in progress. Please wait."


# Initialize session state variables
if "response_text" not in st.session_state:
    st.session_state.response_text = None


if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


if pdf_file and not st.session_state.pdf_processed:
     with st.spinner("Extracting and analyzing your resume..."):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Store extracted text
        response = gu.call_gemini(
            text,
            "You are an AI interviewer. First, you have been provided the extracted data of the PDF. Now organize the data and return it."
        )
        st.session_state.response_text = response.text
        st.session_state.pdf_processed = True
        time.sleep(1)

    

# Show AI response in sidebar
if st.session_state.response_text:
    with st.sidebar:
        st.subheader("Organized Resume")
        st.write(st.session_state.response_text)

# Button to start interview
if st.button("Start Interview"):
    
    response = gu.call_gemini(
        st.session_state.response_text,
        "You are an AI interviewer. you have to start asking the interview questions, don't distract by interview candidates if they are say anything that is not relevent to interview. and ask 10 questions one by one and "
    )
    st.session_state.interview_started = True


# Display interview started section
if st.session_state.interview_started:
    st.success("Interview started!")
    # You can add interview questions here
    

