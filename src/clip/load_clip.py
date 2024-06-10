from image_dataset import image_urls
from load_images import load_images_from_data
from transformers import CLIPModel, CLIPProcessor
from loguru import logger
from load_images import text_classes
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 20))

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images = load_images_from_data(image_urls)

inputs = processor(text=text_classes, images=images, return_tensors="pt", padding=True)

outputs = model(**inputs)
# Extract image and text embeddings
image_embeddings = outputs.image_embeds  # Image embeddings
text_embeddings = outputs.text_embeds  # Text embeddings

logger.info(image_embeddings)
logger.info(text_embeddings)

# Example use case: Calculate cosine similarity between image and text embeddings
from torch.nn.functional import cosine_similarity

# Calculate similarity matrix between all images and text classes
similarity = cosine_similarity(image_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)


def show_grid_results_clip(images, text_classes, probabilities_clip):
    for idx in range(len(images)):
        # show original image
        fig.add_subplot(len(images), 2, 2 * (idx + 1) - 1)
        plt.imshow(images[idx])
        plt.xticks([])
        plt.yticks([])

        # show probabilities
        fig.add_subplot(len(images), 2, 2 * (idx + 1))
        plt.barh(
            range(len(probabilities_clip[0].detach().numpy())),
            probabilities_clip[idx].detach().numpy(),
            tick_label=text_classes,
        )
        plt.xlim(0, 1.0)

        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8
        )

    plt.show()


# Show results (if needed, adjust or replace this function based on how you want to display the results)
show_grid_results_clip(images, text_classes, similarity)
