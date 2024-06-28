def text_to_embedding_transformer(text, model):
    return model.encode(text, convert_to_tensor=True)