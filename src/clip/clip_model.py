from ocr.pytesseract_image_to_text import extract_text_from_image
from image_utils import load_images_from_path, generate_image_metadata
from transformers import CLIPModel, CLIPProcessor
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity
import numpy as np
import os
import torch


class CLIPEmbeddingsModel:

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.fig = plt.figure(figsize=(8, 20))
        self.dataset = None
        self.images = None
        self.embeddings = None
        self.metadata = None

    def load_and_process_dataset(self, image_paths):
        images = load_images_from_path(image_paths)
        self.images = images
        return images

    def process_image(self, image_path, text):
        opened_image = Image.open(image_path)
        processed_image = self.processor(text_classes=text, images=opened_image, return_tensors="pt", padding=True)
        return processed_image

    def generate_dataset_metadata(self, image_paths):
        metadata = generate_image_metadata(image_paths)
        self.metadata = metadata
        logger.info(f"Metadata: {metadata}")
        return metadata

    # store dataset in database
    def store_dataset_locally(self, metadata, embeddings):
        logger.info(f"Storing metadata and embeddings in database")
        # Store metadata and embeddings in database
        logger.info(f"Metadata: {metadata}")
        logger.info(f"Embeddings: {embeddings}")

        # Ensure embeddings are in the right format
        image_embeds = embeddings["image_embeds"]
        text_embeds = embeddings["text_embeds"]

        # Check if the lengths match
        if len(metadata) != image_embeds.size(0) or len(metadata) != text_embeds.size(0):
            raise ValueError("The number of metadata entries must match the number of embeddings")

        combined_data = []
        for i, meta in enumerate(metadata):
            combined_entry = {
                'filename': meta['filename'],
                'path': meta['path'],
                'image_embed': image_embeds[i].tolist(),
                'text_embed': text_embeds[i].tolist()
            }
            combined_data.append(combined_entry)
        # logger.info(combined_data)
        return combined_data

    def generate_dataset_embeddings(self, text_transcriptions):
        inputs = self.processor(text=text_transcriptions, images=self.images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        self.embeddings = self.process_clip_tensors(outputs)

        logger.info(f"Embeddings: {self.embeddings}")
        return outputs

    def process_clip_tensors(self, outputs):
        image_embeds = outputs["image_embeds"]
        text_embeds = outputs["text_embeds"]

        logger.info(f"Image embeddings shape: {image_embeds.shape}")
        logger.info(f"Text embeddings shape: {text_embeds.shape}")

        embeddings = {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds
        }
        return embeddings

    # search for similar images in database (delete duplicates? or keep them?)
    def search_similar_images(self, query):
        # Get text embeddings of class dataset
        text_embeddings = self.embeddings.text_embeds

        # Generate query text embeddings
        query_text_embedding = self.process_and_embedd_query_text(query)

        logger.info(f"Query text embedding shape: {query_text_embedding.shape}")
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")

        # Compute cosine similarity between query text and all text embeddings
        similarities = cosine_similarity(query_text_embedding, text_embeddings, dim=1)
        logger.info(f"Similarity scores: {similarities}")

        # Find the index of the maximum similarity score
        max_similarity_index = torch.argmax(similarities).item()
        max_similarity = similarities[max_similarity_index].item()

        logger.info(f"Max similarity score: {max_similarity} at index: {max_similarity_index}")

        self.__display_similar_image(self.images[max_similarity_index])

    def generate_text_embedding(self, processed_query_text):
        return self.model.get_text_features(**processed_query_text)

    def process_and_embedd_query_text(self, text):
        # Preprocess the text
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Forward pass through the model
        with torch.no_grad():
            # Get the features
            text_embedding = self.generate_text_embedding(inputs)

        # print the embeddings
        return text_embedding

    def __display_similar_image(self, similar_image):
        image = similar_image

        # Convert PIL image to NumPy array
        image_array = np.array(image)

        # Plot the image array using Matplotlib
        plt.imshow(image_array)
        plt.axis('off')  # Hide axis
        plt.show()

    def search_similar_images(self, query):
        text_embeddings = self.embeddings["text_embeds"]

        # Generate query text embeddings
        query_text_embedding = self.process_and_embedd_query_text(query)

        logger.info(f"Query text embedding shape: {query_text_embedding.shape}")
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")

        # Compute cosine similarity between query text and all text embeddings
        similarities = cosine_similarity(query_text_embedding, text_embeddings, dim=1)
        logger.info(f"Similarity scores: {similarities}")

        # Find the index of the maximum similarity score
        max_similarity_index = torch.argmax(similarities).item()
        max_similarity = similarities[max_similarity_index].item()

        logger.info(f"Max similarity score: {max_similarity} at index: {max_similarity_index}")

        # Display the most similar image
        self.__display_similar_image(self.images[max_similarity_index])


# main function to test the class
if __name__ == "__main__":
    clip_model = CLIPEmbeddingsModel()

    base_dir = os.path.dirname(os.path.abspath("../"))

    # Made sure to only take keyframes with short text content
    relative_image_path_1 = os.path.join(base_dir, 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                                         'biology_chapter_3_3-Scene-039-01.jpg')
    relative_image_path_2 = os.path.join(base_dir, 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                                         'biology_chapter_3_3-Scene-097-01.jpg')
    relative_image_path_3 = os.path.join(base_dir, 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                                         'biology_chapter_3_3-Scene-014-01.jpg')

    image_paths = [relative_image_path_1, relative_image_path_2, relative_image_path_3]

    image_dataset = clip_model.load_and_process_dataset(image_paths)

    logger.info(f"Image_dataset: {image_dataset}")

    # from src.ocr.pytesseract_image_to_text import extract_text_from_image

    # Generate OCR Captions

    ocr_extracted_text = []
    for path in image_paths:
        extract_text_from_image(path)
        ocr_extracted_text.append(extract_text_from_image(path))
        logger.info(f"OCR_results: {ocr_extracted_text}")

    # Generate embeddings

    outputs = clip_model.generate_dataset_embeddings(ocr_extracted_text)

    # ----
    clip_model.generate_dataset_metadata(image_paths)

    clip_model.store_dataset_locally(clip_model.metadata, clip_model.embeddings)
    # ----

    # TEST 1: Search for exact similar Text. First, load Test Keyframe Image
    test_image_path = os.path.join(base_dir, 'data', 'raw', 'biology_chapter_3_3', 'extracted_keyframes',
                                   'biology_chapter_3_3-Scene-097-01.jpg')

    test_text_description = extract_text_from_image(test_image_path)

    # Search for similar images in database
    clip_model.search_similar_images(test_text_description)

    # TEST 2: Search for similar images with query text
    query_text = "plasma membrane and stuff going on"

    clip_model.search_similar_images(query_text)

    # ----------
