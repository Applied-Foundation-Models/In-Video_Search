import os

import streamlit as st

from app.app_helper import (
    get_file_mapping,
    load_embeddings,
    load_pickle,
    query_video_data,
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


#### APP FUNCTIONALITY STARTS ####

selected_option = st.selectbox("Select an option:", options)

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
    st.write(f"Data from {file_name}:")
    st.write(st.session_state["data"])


# Use the video embeddings here
# ...

# Text input for prompts
st.session_state.prompt = st.text_input("Enter your prompt here:")

if st.button("Search"):
    st.write("Processing your search")

    # Searchfunctionality:
    (
        col1,
        col2,
    ) = st.columns([0.2, 0.8])

    top_three_results = st.session_state.embedder.retreive_top_3_similar_images(
        st.session_state.prompt
    )
    # # Show top three results in dataframe format:
    selected_result = col1.selectbox("Select a result", top_three_results)
    st.session_state.start_time = query_video_data(
        st.session_state["data"], selected_result, "timestamps"
    )[1]
    col2.text_area(
        "Keyframe summary:",
        value=query_video_data(
            st.session_state["data"], selected_result, "llava_result"
        ),
        height=400,  # Adjust the height as needed
    )

    # Embed a video
st.video(
    "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3_treshhold_5/biology_chapter_3_3_treshhold_5.mp4",
    start_time=st.session_state.start_time,
)
