import requests
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image
from tqdm import tqdm

text_classes = ["giraffe", "zebra", "elephant", "teddybear", "hotdog"]


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


def load_and_preprocess_image(image_path, image_size=224):
    """
    Load an image from the disk and apply preprocessing.
    """
    preprocessing = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocessing(image)


# similarity search: function which compares the input with already stored embeddings
def similarity_search(model, image_embeddings, text_embeddings, query_image, query_text):
    """
    Compare the input image and text with the stored image and text embeddings.

    Args:
    image_embeddings: List of image embeddings
    text_embeddings: List of text embeddings
    query_image: Image to compare
    query_text: Text to compare

    Returns:
    A tuple containing the image and text similarity scores.
    """

    # Preprocess the query image
    query_image = load_and_preprocess_image(query_image)

    # Calculate the query image embedding
    query_image = query_image.unsqueeze(0)
    with torch.no_grad():
        query_image_embedding = model.encode_image(query_image)

    # Calculate the query text embedding
    with torch.no_grad():
        query_text_embedding = model.encode_text(query_text)

    # Calculate the similarity scores
    image_similarity = torch.nn.functional.cosine_similarity(query_image_embedding, image_embeddings, dim=1)
    text_similarity = torch.nn.functional.cosine_similarity(query_text_embedding, text_embeddings, dim=1)

    return image_similarity, text_similarity


def generate_embeddings(model, processor, image_paths, texts, batch_size=32):
    """
    Generate embeddings for a dataset of images and texts using a CLIP model.

    Args:
    model: Pretrained CLIPModel
    processor: Pretrained CLIPProcessor
    image_paths: List of paths to the images
    texts: List of corresponding texts
    batch_size: Number of items to process in one go

    Returns:
    A tuple of tensors containing image and text embeddings.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_images = [load_and_preprocess_image(path) for path in image_paths[i:i + batch_size]]
            batch_texts = texts[i:i + batch_size]

            # Convert list of tensors to a single tensor
            batch_images = torch.stack(batch_images).to(device)

            # Process batch through the model
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True,
                               truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)

            image_embeddings.append(outputs.image_embeds)
            text_embeddings.append(outputs.text_embeds)

    # Concatenate all batches
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    # store the embeddings in a dict
    embeddings = {"image_embeddings": image_embeddings, "text_embeddings": text_embeddings}

    return image_embeddings.cpu(), text_embeddings.cpu(), embeddings
