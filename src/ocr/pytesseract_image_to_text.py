from PIL import Image
import pytesseract
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

print(extract_text_from_image('/Users/haseeb/Desktop/Praktikum/afm-vlm/src/ocr/test.png'))