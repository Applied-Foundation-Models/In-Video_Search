from PIL import Image
import requests

def load_images_from_data(image_urls):
    images = []
    for url in image_urls:
        images.append(Image.open(requests.get(url, stream=True).raw))
    return images