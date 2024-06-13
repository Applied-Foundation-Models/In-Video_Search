import os

import easyocr
from loguru import logger

# Create an OCR reader object
reader = easyocr.Reader(["en"])

base_dir = os.path.dirname(os.path.abspath(__file__))

relative_image_path = os.path.join(base_dir, "ocr", "test.png")

result = reader.readtext(relative_image_path)

for detection in result:
    logger.info(detection[1])
