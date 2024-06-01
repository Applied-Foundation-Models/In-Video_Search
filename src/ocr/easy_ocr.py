import easyocr

# Create an OCR reader object
reader = easyocr.Reader(['en'])

# Read text from an image
result = reader.readtext('/Users/haseeb/Desktop/Praktikum/afm-vlm/src/ocr/test.png')

for detection in result:
    print(detection[1])