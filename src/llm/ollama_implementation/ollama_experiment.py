import argparse
import base64
import json
from io import BytesIO

import ollama
import requests
from PIL import Image

"""
the following relies on ollama which can be used to run LLMs locally it is optimized for usage on Apple MPS

downlaod and install here: https://www.ollama.com/
"""


url = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json",
}

conversation_history = []


def generate_response(slide_content, transcription):
    # conversation_history.append(prompt)

    # full_prompt = "\n".join(conversation_history)

    # Define the prompt with placeholders
    prompt_template = """
    You are a highly intelligent assistant. Your task is to summarize lecture content that includes text extracted from slides and spoken content transcribed from the lecture. Provide a concise summary that highlights the main points discussed on the slides and the key topics spoken by the lecturer in order to make the lecture queryable.

    ### Slide Content:
    {slide_content}

    ### Transcription:
    {transcription}

    ### Summary:
    - **Slide Content Summary:**
    - [Summarize the main points from the slides]

    - **Spoken Content Summary:**
    - [Summarize the key topics discussed by the lecturer]
    """

    # Fill the placeholders
    prompt = prompt_template.format(
        slide_content=slide_content, transcription=transcription
    )

    data = {
        "model": "llama3",
        "stream": False,
        "prompt": prompt,
        "options": {"seed": 1, "temperature": 0.9, "num_predict": 77},
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

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
    # Open the image, convert to the specified format, and save to a buffer
    with Image.open(image_path) as image:
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        # Encode the image bytes to base64
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64


def generate_caption_using_llava(image_file):
    # print(f"\nProcessing {image_file}\n")
    # with Image.open(image_file) as img:
    #     with BytesIO() as buffer:
    #         img.save(buffer, format="PNG")
    #         image_bytes = buffer.getvalue()
    # data = {
    #     "model": "llava",
    #     "stream": False,
    #     "prompt": prompt,
    #     "images": [image_bytes],
    # }

    # response = requests.post(url, headers=headers, data=json.dumps(data))
    # print(response)
    image = image_to_base64(image_file)
    res = ollama.chat(
        model="llava",
        messages=[
            {"role": "user", "content": "Describe this image:", "images": [image]}
        ],
    )

    print(res["message"]["content"])
    return res["message"]["content"]

    # full_response = ''
    # # Generate a description of the image
    # for response in generate(model='llava:13b-v1.6',
    #                          prompt='describe this image and make sure to include anything notable about it (include text you see in the image):',
    #                          images=[image_bytes],
    #                          stream=False):
    #     # Print the response to the console and add it to the full response
    #     print(response['response'], end='', flush=True)
    #     full_response += response['response']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello, how can I assist you?")
    args = parser.parse_args()

    prompt = args.prompt
    # response = generate_response(
    #     slide_content="My text on my slide", transcription="my transcription"
    # )
    # print(response)

    text = generate_caption_using_llava(
        "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3/extracted_keyframes/biology_chapter_3_3-Scene-032-01.jpg"
    )
    print(text)
