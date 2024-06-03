from image_dataset import image_urls
from load_images import load_images_from_data
from show_grid_results import show_grid_results_clip
from transformers import CLIPModel, CLIPProcessor

text_classes = ["giraffe", "zebra", "elephant", "teddybear", "hotdog"]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images = load_images_from_data(image_urls)

inputs = processor(text=text_classes, images=images, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probabilities_clip = logits_per_image.softmax(
    dim=1
)  # we can take the softmax to get the label probabilities


show_grid_results_clip(images, text_classes, probabilities_clip)
