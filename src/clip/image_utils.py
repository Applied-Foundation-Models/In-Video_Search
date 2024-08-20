import os

from loguru import logger
from PIL import Image


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
            "filename": filename,
            "path": path,
        }
        metadata.append(metadata_entry)

    logger.info(f"Created metadata for {len(metadata)} images")
    logger.info(f"Metadata: {metadata}")

    return metadata
