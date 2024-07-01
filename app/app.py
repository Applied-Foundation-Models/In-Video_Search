import os

import streamlit as st
from app_helper import get_file_mapping, load_pickle

st.set_page_config(
    page_title=" Welcome to Video Summarization - Your Study helper!",
    page_icon="👋",
    layout="centered",
)

# Get the current path of the project
project_path = os.getcwd()

options = list(get_file_mapping().keys())

if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "start_time" not in st.session_state:
    st.session_state.start_time = 0

# Print the current path


st.title("Welcome to Video Summarization - Your Study helper!")

with st.sidebar:
    # Display username in the sidebar
    st.title("Welcome to Video Summarization - Your Study helper!")

    st.markdown(
        """
    ---
    Created with ❤️ by EduSummarize.
    """
    )

selected_option = st.selectbox("Select an option:", options)

# Initialize session state for data
if "data" not in st.session_state:
    st.session_state["data"] = None

if selected_option:
    file_mapping = get_file_mapping()
    file_name = file_mapping[selected_option]

    if st.button("Retrieve video"):
        st.session_state["data"] = load_pickle(os.path.join(project_path, file_name))

# Display data if available in session state
if st.session_state["data"] is not None:
    st.write(f"Data from {file_name}:")
    st.write(st.session_state["data"])


# Use the video embeddings here
# ...

# Text input for prompts
st.session_state.prompt = st.text_input("Enter your prompt here:")

if st.session_state.start_time != "":
    st.write("Processing your search")

    # Doing search functionality:


# Embed a video
st.video(
    "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3_treshhold_5/biology_chapter_3_3_treshhold_5.mp4",
    start_time=st.session_state.start_time,
)
