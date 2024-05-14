import torch
from PIL import Image
import requests
from transformers import ViTFeatureExtractor, GPT2LMHeadModel, GPT2Tokenizer
from io import BytesIO

# Load the feature extractor and models
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def prepare_image(image):
    """Prepare the image for model prediction using the feature extractor."""
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values']

def generate_caption(pixel_values):
    """Generate a caption for the image."""
    # This example assumes you have some method to convert image features to text prompts
    text_prompt = "Describe the image:"  # Placeholder for the image-to-text conversion logic
    inputs = tokenizer.encode(text_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

# Example usage
image_url = "https://example.com/your-image.jpg"
image = load_image_from_url(image_url)
pixel_values = prepare_image(image)
caption = generate_caption(pixel_values)
print("Generated Caption:", caption)
