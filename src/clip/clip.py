import torch
import clip


class CLIP:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = self.__load_model()

    def __load_model(self) -> tuple:
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    # load tokenizer
    def tokenize_text(self, text_descriptions) -> torch.Tensor:
        tokenized_text = clip.tokenize(text_descriptions).to(self.device)
        return tokenized_text

    def generate_dataset_embeddings(self, image_dataset, texts_descriptions) -> tuple:
        dataset_images_preprocessed = torch.cat([self.preprocess(image).unsqueeze(0) for image in image_dataset],
                                                dim=0).to(
            self.device)
        text_inputs = clip.tokenize(texts_descriptions).to(self.device)

        with torch.no_grad():
            dataset_image_embeddings = self.model.encode_image(dataset_images_preprocessed)
            dataset_text_embeddings = self.model.encode_text(text_inputs)

        return dataset_image_embeddings, dataset_text_embeddings

    def similarity_search(self, image_embeddings, text_embeddings, query_image, query_text):
        query_image = self.preprocess(query_image).unsqueeze(0).to(self.device)
        query_image_embedding = self.model.encode_image(query_image)

        query_text_embedding = self.model.encode_text(query_text)

        image_similarity = torch.nn.functional.cosine_similarity(query_image_embedding, image_embeddings, dim=1)
        text_similarity = torch.nn.functional.cosine_similarity(query_text_embedding, text_embeddings, dim=1)

        return image_similarity, text_similarity
