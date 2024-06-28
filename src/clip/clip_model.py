from src.ocr.pytesseract_image_to_text import extract_text_from_image
from src.clip.image_utils import load_images_from_path, generate_image_metadata
from src.text_embedder.embedder import text_to_embedding_transformer
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
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
        self.text_embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.fig = plt.figure(figsize=(8, 20))
        self.dataset = None
        self.images = None
        self.embeddings = None
        self.text_embeddings = None
        self.metadata = None
        self.img_paths = None
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

    def generate_dataset_embeddings_standard_tokenizer(self, text):
        model = self.text_embedder
        query_text_embedding = text_to_embedding_transformer(text, model)
        self.text_embeddings = query_text_embedding
        logger.info(f"Text embeddings with standard tokenizer: {self.text_embeddings}")

    def generate_image_embeddings(self, text_transcription, image):
        inputs = self.processor(
            text=text_transcription,
            images=image,
            return_tensors="pt",
            max_length=77,
            truncation=True,
        )

        # logger.info(f"Inputs id shape: {inputs['input_ids'].shape}")
        # logger.info(f"Positions id shape: {inputs['position_ids'].shape}")
        outputs = self.model(**inputs)
        self.embeddings = self.process_clip_tensors(outputs)
        embeddings = outputs
        return embeddings

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
        text_embeddings = self.text_embeddings

        # Generate query text embeddings
        # query_text_embedding = self.process_and_embedd_query_text(query)
        model = self.text_embedder

        query_text_embedding = text_to_embedding_transformer(query, model)

        logger.info(f"Query text embedding shape: {query_text_embedding.shape}")
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")

        # Compute cosine similarity between query text and all text embeddings
        similarities = cosine_similarity(query_text_embedding, text_embeddings, dim=1)

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

    def search_similar_images_top_3(self, query, gt):
        # text_embeddings = self.embeddings["text_embeds"]
        text_embeddings = self.text_embeddings

        # Generate query text embeddings
        model = self.text_embedder

        query_text_embedding = text_to_embedding_transformer(query, model)

        logger.info(f"Query text embedding shape: {query_text_embedding.shape}")
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")

        # Compute cosine similarity between query text and all text embeddings
        similarities = cosine_similarity(query_text_embedding, text_embeddings, dim=1)
        logger.info(f"Similarity scores: {similarities}")

        indices = torch.topk(similarities, 3)

        logger.info(f"Top 3 Similarity scores: {indices} - GT: {gt}")

        # Display the most similar image
        # dispolay top 3 images
        for i in range(3):
            max_similarity_index = indices.indices[i].item()
            logger.info(f"Max similarity index: {max_similarity_index}")
            self.__display_similar_image(self.images[max_similarity_index])

    def search_similar_image_revisited(self, query):
        # Get text embeddings of class dataset
        text_embeddings = self.text_embeddings
        logger.info(f"Text embeddings {text_embeddings.shape}")

        # Generate query text embeddings
        query_text_embedding = self.process_and_embedd_query_text(query)

        # Compute cosine similarity between query text and all text embeddings
        similarities = cosine_similarity(query_text_embedding, text_embeddings, dim=1)
        logger.info(f"Similarity scores: {similarities}")

        logger.info(f"Similarity scores: {similarities}")

        max_similarity_index = torch.argmax(similarities).item()
        logger.info(f"Max similarity index: {max_similarity_index}")

        # get image path
        image_path = self.img_paths[max_similarity_index]
        logger.info(f"Image path: {image_path}")
