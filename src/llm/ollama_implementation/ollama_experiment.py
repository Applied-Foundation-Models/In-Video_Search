import argparse
import base64
import json
from io import BytesIO

import requests
from PIL import Image

"""
the following relies on ollama which can be used to run LLMs locally it is optimized for usage on Apple MPS

downlaod and install here: https://www.ollama.com/
"""


URL = "http://localhost:11434/api/generate"

HEADERS = {
    "Content-Type": "application/json",
}

conversation_history = []


def prompt_llm_summary(slide_content, transcription, llava_output):
    """
    Generate a prompt for summarizing lecture content.

    Args:
        slide_content (str): The text extracted from the slides.
        transcription (str): The spoken content transcribed from the lecture.
        llava_output (str): Visual insights about slide composition and information on figures displayed in the slide.

    Returns:
        str: The generated summary of the lecture content.

    Raises:
        None

    Example:
        summary = prompt_llm_summary(slide_content, transcription, llava_output)
    """

    prompt_template = """
    Task: Summarize lecture content that includes text extracted from slides, spoken content transcribed
    from the lecture, and LLAVA output. This will later be used to query a database to find relevant information. So make sure to include the most important keywords

    ### Slide Content:
    {slide_content} (Main points: [list key takeaways])

    ### Transcription:
    {transcription} (Key topics: [list main ideas])

    ### LLAVA Output:
    {llava_output} (Visual insights about slide composure and information on figures displayed in the slide: [summarize key findings])

    ### Summary:
    - **Slide Summary:** Please summarize the main points discussed on the slides in
    approximately 100 words and summarize the key topics spoken by the lecturer, highlighting the most
    important ideas and concepts.

    **Queryable Information:** tags, categories, or specific concepts (e.g., AI ethics, natural
    language processing, deep learning)
    """

    # Fill the placeholders
    summary = prompt_template.format(
        slide_content=slide_content,
        transcription=transcription,
        llava_output=llava_output,
    )

    data = {
        "model": "llama3",
        "stream": False,
        "prompt": summary,
        "options": {"seed": 1, "temperature": 0.2},
    }

    response = requests.post(URL, headers=HEADERS, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        conversation_history.append(actual_response)
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None


def image_to_base64(image_path, format="JPEG"):
    """
    Convert an image file to base64 encoding.

    Parameters:
    image_path (str): The path to the image file.
    format (str): The format of the image file. Default is "JPEG".

    Returns:
    str: The base64 encoded string representation of the image.
    """
    # Open the image, convert to the specified format, and save to a buffer
    with Image.open(image_path) as image:
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        # Encode the image bytes to base64
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64


def generate_caption_using_llava(image_file):
    """
    Generates a caption using the LLAVA model based on an input image.

    Args:
        image_file (str): The file path of the input image.

    Returns:
        str: The generated caption for the input image.
    """

    image = image_to_base64(image_file)

    data = {
        "model": "llava",
        "stream": False,
        "prompt": "Describe the following slide taken from an academic lecture. Structure your response into the discovered figures as well as headlines and summarize the concepts explained on the slide :",
        "options": {"seed": 1, "temperature": 0.2},
        "images": [image],
    }

    response = requests.post(URL, headers=HEADERS, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        conversation_history.append(actual_response)
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run either function 1 or 2")
    parser.add_argument(
        "--prompt_llm_summary",
        action="store_true",
        default=False,
        help="Run function 1",
    )
    parser.add_argument(
        "--llava_captioning", action="store_true", default=False, help="Run function 2"
    )

    args = parser.parse_args()

    if args.prompt_llm_summary:
        response = prompt_llm_summary(
            slide_content="My text on my slide",
            transcription="my transcription",
            llava_output="/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3/extracted_keyframes/biology_chapter_3_3-Scene-032-01.jpg",
        )
        print(response)

    elif args.llava_captioning:
        response = generate_caption_using_llava(
            "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3/extracted_keyframes/biology_chapter_3_3-Scene-032-01.jpg"
        )
        print(response)
