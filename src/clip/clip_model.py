import torch
from .load_images import load_images_from_data, text_classes, load_images_from_path
from transformers import CLIPModel, CLIPProcessor
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity

from ..ocr.pytesseract_image_to_text import extract_text_from_image


class CLIPEmbeddingsModel:

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.fig = plt.figure(figsize=(8, 20))
        self.images = None
        self.embeddings = None

    def load_and_process_dataset(self, image_paths):
        images = load_images_from_path(image_paths)
        self.images = images
        return images

    def process_image(self, image_path, text):
        test_image = Image.open(image_path)
        return self.processor(text_classes=text, images=test_image, return_tensors="pt", padding=True)

    def generate_dataset_embeddings(self, text_transcriptions):
        inputs = self.processor(text=text_transcriptions, images=self.images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        self.embeddings = outputs
        logger.info(outputs)
        return outputs

    def search_similar_images(self, image_embeddings, text_embeddings):
        similarity = cosine_similarity(image_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)
        return similarity

    # search for similar images in database with query text
    def search_similar_images_with_text(self, query_text):
        # get text embeddings of class dataset
        text_embeddings = self.embeddings.text_embeds
        # generate query text embeddings
        query_text_embedding = self.model.encode_text(query_text)
        text_similarity = cosine_similarity(query_text_embedding, text_embeddings, dim=1)
        return text_similarity


# Example use case

# instance
clip_model = CLIPEmbeddingsModel()

# load images
import os
from loguru import logger

base_dir = os.path.dirname(os.path.abspath("."))

relative_image_path_1 = os.path.join(base_dir, 'afm-vlm', 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                                     'biology_chapter_3_3-Scene-039-01.jpg')
relative_image_path_2 = os.path.join(base_dir, 'afm-vlm', 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                                     'biology_chapter_3_3-Scene-099-01.jpg')
relative_image_path_3 = os.path.join(base_dir, 'afm-vlm', 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                                     'biology_chapter_3_3-Scene-014-01.jpg')

image_paths = [relative_image_path_1, relative_image_path_2, relative_image_path_3]

image_dataset = clip_model.load_and_process_dataset(image_paths)

logger.info(f"Image_dataset: {image_dataset}")

# Generate OCR Captions

ocr_extracted_text = []
for path in image_paths:
    extract_text_from_image(path)
    ocr_extracted_text.append(extract_text_from_image(path))
    logger.info(f"OCR_results: {ocr_extracted_text}")

# Generate embeddings

outputs = clip_model.generate_dataset_embeddings(ocr_extracted_text)

# Load Test Keyframe Image

test_image_path = os.path.join(base_dir, 'afm-vlm', 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                               'biology_chapter_3_3-Scene-099-01.jpg')

test_text_description = extract_text_from_image(test_image_path)

test_image_preprocessed = clip_model.process_image(test_image_path, test_text_description)

# Search for similar images in database

similarity = clip_model.search_similar_images(outputs.image_embeds, outputs.text_embeds)

# Search for similar images with query text

text_similarity = clip_model.search_similar_images_with_text(test_text_description)
