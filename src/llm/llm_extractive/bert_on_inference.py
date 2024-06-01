# bert_summarizer.py

import argparse
from transformers import BertModel, BertTokenizer
import torch
import nltk
from nltk.tokenize import sent_tokenize

def summarize(text):
    nltk.download('punkt')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Split text into sentences
    sentences = sent_tokenize(text)

    # Process each sentence for BERT
    embeddings = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        outputs = model(**inputs)
        sent_embedding = outputs.last_hidden_state.mean(1)
        embeddings.append(sent_embedding)

    # Calculate mean across embeddings to simulate sentence importance
    embeddings_tensor = torch.cat(embeddings, dim=0)
    sentence_scores = torch.mean(embeddings_tensor, dim=1)
    top_sentence = torch.argmax(sentence_scores)

    return sentences[top_sentence]

def main():
    parser = argparse.ArgumentParser(description='BERT Extractive Summarization')
    parser.add_argument('text', type=str, help='Text to summarize')
    args = parser.parse_args()

    summary = summarize(args.text)
    print("Summary:", summary)

if __name__ == '__main__':
    main()
