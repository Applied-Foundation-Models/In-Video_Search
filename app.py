import os

import streamlit as st
from loguru import logger

from app.app_helper import (
    create_tensor_and_mapping,
    get_file_mapping,
    load_pickle,
    retrieve_keys_from_indices,
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

if "video_url" not in st.session_state:
    st.session_state.video_url = (
        "data/raw/biology_chapter_3_3_treshhold_5/biology_chapter_3_3_treshhold_5.mp4"
    )
if "keyframe_summary" not in st.session_state:
    st.session_state.keyframe_summary = ""
# Print the current path
if "mapping" not in st.session_state:
    st.session_state.mapping = {}


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
    pickle_file = file_mapping[selected_option]["pickle_file"]
    video_url = file_mapping[selected_option]["video_url"]
    st.session_state["data"] = load_pickle(os.path.join(project_path, pickle_file))
    st.session_state.embedder.text_embeddings, st.session_state.mapping = (
        create_tensor_and_mapping(st.session_state["data"], "extensive_text_embedding")
    )
    # Set session state variable for video url to be used in video player:
    st.session_state["video_url"] = os.path.join(project_path, video_url)


st.header(f"Recap Lecture: {selected_option}")

st.video(
    st.session_state.video_url,
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
    top_three_results_index = st.session_state.embedder.retreive_top_3_similar_images(
        st.session_state.prompt
    )
    top_three_results = retrieve_keys_from_indices(
        top_three_results_index, st.session_state.mapping
    )
    print(f"Top three results: {top_three_results}")
    # Map the three results to the corresponding keys: (we need to iterate over the values and find the corresponding key frma the value )

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
    logger.info("Updating selection:")
    selected_result = col1.selectbox(
        "Select a result",
        st.session_state.top_three_results,
        key="selected_result",
        on_change=update_selection,
    )
    logger.info(f"Selected result: {selected_result}")

    # Display selected keyframe summary in text area in col2
    col2.text_area(
        "Keyframe summary:",
        value=st.session_state.keyframe_summary,
        height=400,  # Adjust the height as needed
    )
