import os
import pytesseract
from PIL import Image
from loguru import logger


def extract_text_from_image(image_path):
    # Define the base directory relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(base_dir, image_path)
    image = Image.open(full_image_path)
    text = pytesseract.image_to_string(image)
    logger.info(f"Extracted text: {text}")
    return text
