import os

import pytesseract
from loguru import logger
from PIL import Image

# Define the base directory relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))


def extract_text_from_image_deprecated(image_path):
    full_image_path = os.path.join(base_dir, image_path)
    image = Image.open(full_image_path)
    text = pytesseract.image_to_string(image)
    return text


# Specify the relative path to the image from the script's location
relative_image_path = "test.png"
logger.info(f"Extracted text: {extract_text_from_image_deprecated(relative_image_path)}")

def extract_text_from_image(image_path):
    # Define the base directory relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(base_dir, image_path)
    image = Image.open(full_image_path)
    text = pytesseract.image_to_string(image)
    logger.info(f"Extracted text: {text}")
    return text