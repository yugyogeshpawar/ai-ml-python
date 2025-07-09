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

if "interview_history" not in st.session_state:
    st.session_state.interview_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "current_question" not in st.session_state:
    st.session_state.current_question = ""


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
            "You are an AI interviewer. First, you have been provided the extracted data of the PDF. Now organize the data and return it. if that is not a resume texts please reply error msg. reply in markdown for error."
        )
        st.session_state.response_text = response.text
        st.session_state.pdf_processed = True
        time.sleep(1)

    

# Show AI response in sidebar
if st.session_state.response_text:
    with st.sidebar:
        st.subheader("Organized Resume")
        st.markdown(st.session_state.response_text)

# Button to start interview
if st.button("Start Interview"):
    with st.spinner("Extracting and analyzing your resume..."):
        
        response = gu.call_gemini(
            st.session_state.response_text,
            "You are an AI interviewer. you have to start asking the technical interview questions in short and based on resume profile (designation), don't distract by interview candidates if they are say anything that is not relevent to interview. and ask 10 questions one by one."
        )
        
        st.session_state.current_question = response.text
        st.session_state.interview_history.append({"role": "AI", "text": response.text})
        st.session_state.interview_started = True
        st.session_state.question_count = 1

if st.session_state.interview_started and st.session_state.current_question:
    st.markdown(f"**Question {st.session_state.question_count} :** {st.session_state.current_question}")
    user_answer = st.text_input("Your answer:", key=st.session_state.question_count)

    if st.button("Submit Answer"):
        st.session_state.interview_history.append({"role":"User","text":user_answer})

        st.subheader("History")
        conversation = ""
        for x in st.session_state.interview_history:
            role = x["role"]
            text = x["text"]
            conversation += f"{role}: {text}\n"
        
        response = gu.call_gemini(conversation, "You are an AI interviewer. Continue the interview with the next technical question and in short . Do not repeat any previous question. Limit to 10 questions total. If it's the last question, conclude the interview. and give the marks out of 100.")

        st.session_state.interview_history.append({"role":"AI","text":response.text})
        st.session_state.current_question = response.text
        st.session_state.question_count += 1

        if st.session_state.question_count > 10:
            st.success("Interview completed!")



        


