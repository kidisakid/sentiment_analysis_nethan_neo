# Using bert for sentiment analysis
# https://huggingface.co/transformers/v3.0.2/model_doc/

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sentiment_analysis(texts):
    pipeline = pipeline(
        task='sentiment-analysis',
        # Load the model and tokenizer from Hugging Face
        model='distilbert-base-uncased-finetuned-sst-2-english'
    )

    for text in texts:
        result = pipeline(text)
        # Print the result magnitude/score as negative, neutral, or positive
        # using score thresholds: negative (<0.4), neutral (0.4-0.6), positive (>0.6)

        print(f"Text: {text}\nSentiment: {result}\n")

    return result
    
