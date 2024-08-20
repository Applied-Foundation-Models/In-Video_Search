# bart_summarizer.py

import argparse

from transformers import BartForConditionalGeneration, BartTokenizer


def summarize(text):
    """
    Summarizes the given text using the BART model.

    Args:
        text (str): The input text to be summarized.

    Returns:
        str: The generated summary of the input text.
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    # Tokenize and prepare the input
    inputs = tokenizer(
        [text],
        max_length=1024,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="BART Abstractive Summarization")
    parser.add_argument("text", type=str, help="Text to summarize")
    args = parser.parse_args()

    summary = summarize(args.text)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
