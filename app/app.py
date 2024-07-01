import os
import pickle

import streamlit as st

st.set_page_config(
    page_title=" Welcome to Video Summarization - Your Study helper!",
    page_icon="👋",
    layout="centered",
)

# Get the current path of the project
project_path = os.getcwd()
# Print the current path


def main():
    st.title("Welcome to Video Summarization - Your Study helper!")

    with st.sidebar:
        # Display username in the sidebar
        st.title("Welcome to Video Summarization - Your Study helper!")

        st.markdown(
            """
        ---
        Created with ❤️ by UComply.
        """
        )

    # Load pickle file with video embeddings
    with open(os.path.join(project_path, "data.pickle"), "rb") as f:
        video_embeddings = pickle.load(f)
    print(video_embeddings)

    # Use the video embeddings here
    # ...

    # Text input for prompts
    prompt = st.text_input("Enter your prompt here:")

    if prompt:
        st.write(f"You entered: {prompt}")

    # Embed a video
    st.video(
        "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3_treshhold_5/biology_chapter_3_3_treshhold_5.mp4",
        start_time=300,
    )


if __name__ == "__main__":
    main()
