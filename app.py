import os

import streamlit as st

from app.app_helper import (
    get_file_mapping,
    load_embeddings,
    load_pickle,
    update_selection,
)
from src.text_embedder.embedder import EmbeddingsModel

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
if "data" not in st.session_state:
    st.session_state["data"] = None

if "embedder" not in st.session_state:
    st.session_state["embedder"] = EmbeddingsModel()
if "top_three_results" not in st.session_state:
    st.session_state.top_three_results = []


# Print the current path


st.title("Welcome to Video Summarization!")

with st.sidebar:
    # Display username in the sidebar
    st.title("Welcome to Video Summarization - Your Study helper!")

    st.markdown(
        """
    ---
    Created with ❤️ by EduSummarize.
    """
    )


#### APP FUNCTIONALITY STARTS ####

selected_option = st.selectbox("Select a Lecture to study:", options)

# Initialize session state for data


if selected_option:
    file_mapping = get_file_mapping()
    file_name = file_mapping[selected_option]
    st.session_state["data"] = load_pickle(os.path.join(project_path, file_name))
    st.session_state.embedder.text_embeddings = load_embeddings(
        st.session_state["data"], "clip_text_embedding"
    )


# Display data if available in session state
if st.session_state["data"] is not None:
    # st.write(st.session_state["data"])
    st.write("Lecture loaded successfully!")

st.header("Video Search:")

st.video(
    "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3_treshhold_5/biology_chapter_3_3_treshhold_5.mp4",
    start_time=st.session_state.start_time,
)
st.markdown(
    """
    <style>
    .form-container {
        display: flex;
        height: 3.4vh;
        align-items: center;  /* Center vertically */
        justify-content: center;  /* Center horizontally */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create columns for input and button
input_col, button_col = st.columns([0.80, 0.20])  # Adjust the proportions as needed

with input_col:
    st.session_state.prompt = st.text_input("Enter your prompt here:")

with button_col:
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    search_button_pressed = st.button(
        "🔮", use_container_width=True
    )  # Using an emoji for the button

    st.markdown("</div>", unsafe_allow_html=True)

if search_button_pressed:
    top_three_results = st.session_state.embedder.retreive_top_3_similar_images(
        st.session_state.prompt
    )

    st.session_state.top_three_results = top_three_results
    st.session_state.selected_result = top_three_results[
        0
    ]  # Default to the first result
    update_selection()
st.divider()
st.header("Results:")
col1, col2 = st.columns([0.2, 0.8])

# Replace table with selectbox for top three results
if st.session_state.top_three_results:
    selected_result = col1.selectbox(
        "Select a result",
        st.session_state.top_three_results,
        key="selected_result",
        on_change=update_selection,
    )

    # Display selected keyframe summary in text area in col2
    col2.text_area(
        "Keyframe summary:",
        value=st.session_state.keyframe_summary,
        height=400,  # Adjust the height as needed
    )
