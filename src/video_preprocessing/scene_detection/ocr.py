import argparse
import os
import pickle
from collections import OrderedDict

import cv2
import easyocr
import numpy as np
import pytesseract
from pytesseract import Output

parser = argparse.ArgumentParser()
parser.add_argument(
    "-base_path",
    "--base_path",
    help="base folder path of dataset ",
    default="/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/bio-4_l1_ch1/video_chunks/1_7/scenes",
)
args = parser.parse_args()

BASE_PATH = args.base_path


def get_OCR_pytesseract(img_path):
    image = cv2.imread(img_path)
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    # get surrounding median color
    image_2 = image.copy()
    unique, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    image_2[:, :, 0], image_2[:, :, 1], image_2[:, :, 2] = unique[np.argmax(counts)]

    kf_dict = OrderedDict()

    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]

        text = results["text"][i]

        conf = int(results["conf"][i])
        text = text.strip()
        if conf > 70 and len(text) > 0:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2
            )
            kf_dict[text] = [x, y, w, h]

    return kf_dict


def get_ocr_easyocr(img_path):
    # specify once which language we need
    reader = easyocr.Reader(["en"])

    result = reader.readtext(img_path, paragraph=True)

    return result


for dirpath, subdirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(dirpath, file)
            ocr_dict = get_OCR_pytesseract(img_path)
            # ocr_result = get_ocr_easyocr(img_path)

            file = file.replace(".jpg", ".pickle")
            dest_path = os.path.join(BASE_PATH, file)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as file_:
                pickle.dump(ocr_dict, file_, -1)


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as file:
        ocr_dict = pickle.load(file)
        print(ocr_dict)
