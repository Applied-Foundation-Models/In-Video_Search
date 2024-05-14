from transformers import GitVisionConfig, GitVisionModel, pipeline
from PIL import Image
import requests
from transformers import AutoProcessor, GitVisionModel


# main function
def main():
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = GitVisionModel.from_pretrained("microsoft/git-base")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state


def git_model():
    from transformers import AutoProcessor, AutoModel
    import requests
    from PIL import Image

    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModel.from_pretrained("microsoft/git-base")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    text = "this is an image of two cats"

    inputs = processor(text, images=image, return_tensors="pt")

    outputs = model(**inputs)
    print(outputs.last_hidden_state)
    last_hidden_state = outputs.last_hidden_state


def model_v2():
    from transformers import AutoProcessor, AutoModelForSequenceClassification

    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/git-base")

    text = "Fixes bug causing server crash on heavy load."
    inputs = processor(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)
    print(predictions)


def model_v3():
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    code_snippet = "import numpy as np\nnp."
    inputs = processor(code_snippet, return_tensors="pt")
    outputs = model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]
    predicted_token_id = next_token_logits.argmax(-1)
    predicted_token = processor.decode(predicted_token_id)
    print(predicted_token)


# Entry point of the script
# call main
if __name__ == "__main__":
    model_v2()
