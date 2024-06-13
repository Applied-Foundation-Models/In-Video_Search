import requests
from PIL import Image
import os
from loguru import logger


def load_images_from_data(image_urls):
    images = []
    for url in image_urls:
        images.append(Image.open(requests.get(url, stream=True).raw))
    return images


def load_images_from_path(image_paths):
    images = []
    for path in image_paths:
        images.append(Image.open(path))
    return images


def generate_image_metadata(image_paths):
    metadata = []
    for path in image_paths:
        filename = os.path.basename(path)
        metadata_entry = {
            'filename': filename,
            'path': path,
        }
        metadata.append(metadata_entry)

    logger.info(f"Created metadata for {len(metadata)} images")
    logger.info(f"Metadata: {metadata}")

    return metadata
