import os

import pytesseract
from loguru import logger
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
logger.info(dir_path)


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


# @ Azza this fails on import as we have custom paths in here
# logger.info(
#     extract_text_from_image("/Users/haseeb/Desktop/Praktikum/afm-vlm/src/ocr/test.png")
# )
