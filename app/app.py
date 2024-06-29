import streamlit as st


def main():
    st.title("Streamlit App with Text Input and Embedded Video")

    # Text input for prompts
    prompt = st.text_input("Enter your prompt here:")

    if prompt:
        st.write(f"You entered: {prompt}")

    # Embed a video
    st.video(
        "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3_treshhold_5/biology_chapter_3_3_treshhold_5.mp4",
        start_time=30,
    )


if __name__ == "__main__":
    main()
