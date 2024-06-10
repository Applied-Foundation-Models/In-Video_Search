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
    - **Slide Summary:** Combine and understand all this information from slide content and transcription, and llava output. Give a combined final summary of 77 characters/tokens or less of all the overall context and main points discussed on
    the slides in and summarize the key topics spoken by the lecturer, highlighting the most important ideas and concepts. You can only include keyphrases/ important words. Do not give individual summaries of each transcription, llava output, and slide content. 
    The output should only contain the summary and no other text. 
    
    For example: 
    ### Slide Content: Nutrients: Micronutrients - Vitamins: organic substances — Usually function as coenzymes - Help to speed up body’s chemical reactions — Only vitamin D can be synthesized in the body 
    - Sunlight required - Supplementation in areas of low sunlight — Other vitamins are supplied by foods 
    
    ### Transcription: Vitaminins are first type of micronutrient, organic substances, most of which the body cannot synthesize on its own.  Once in the body, most vitamins function as what's called a co-enzyme.  
    A co-enzyme is a molecule that helps enzymes and thus helps to speed up that enzymes work in completing a body's chemical reaction.  Vitamin deficiencies can affect every cell in the body because many different enzymes, all requiring the same vitamin, are involved in numerous bodily functions. 
    Vitaminins can even help protect the body against cancer and heart disease and even slows the aging process.  Vitamin D, which is also called calcetriol, is the only vitamin that ourselves and our body can synthesize on their own. 
    But there's a catch, sunlight is required for that process, so people living in climates with little sunlight can more easily develop a deficiency of vitamin D in those areas than those that live in places where there's plenty of sunlight.  Healthcare providers may recommend vitamin D supplements to these people.  
    You might also see that milk often comes supplemented with vitamin D.  All other vitamins then must be supplied by the foods that we eat.
    
    ### LLAVA Output: The slide appears to be from an academic lecture discussing the topic of Microcronutrients.
    
    ### Summary: Vitamins: essential micronutrients, coenzymes, speed reactions, only vitamin D synthesized with sunlight, deficiency impacts, supplements needed, diet source.
    
    There should be no other output except for the summary. Do not include " Here is your summary", just the output summary. 
    
    Sample output: Vitamins: essential micronutrients, coenzymes, speed reactions, only vitamin D synthesized with sunlight, deficiency impacts, supplements needed, diet source.
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

    # if args.prompt_llm_summary:
    response = prompt_llm_summary(
        slide_content="My text on my slide",
        transcription="my transcription",
        llava_output="/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3/extracted_keyframes/biology_chapter_3_3-Scene-032-01.jpg",
    )
    print(response)

    # elif args.llava_captioning:
    #     response = generate_caption_using_llava(
    #         "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3/extracted_keyframes/biology_chapter_3_3-Scene-032-01.jpg"
    #     )
    #     print(response)
