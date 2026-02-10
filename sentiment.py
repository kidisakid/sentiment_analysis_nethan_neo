# Using bert for sentiment analysis
# https://huggingface.co/transformers/v3.0.2/model_doc/

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pipeline = pipeline(
    task='sentiment-analysis',
    # Load the model and tokenizer from Hugging Face
    model='nlptown/bert-base-multilingual-uncased-sentiment',

)

texts = [
    "I love this product! It works great and exceeded my expectations.",
    "This is the worst experience I've ever had. I'm very disappointed.",
    "The movie was okay, not the best but not the worst either.",
    "I am extremely happy with the service. The staff was friendly and helpful.",
    "The food was terrible. I will never eat here again."
    "Good staff, but the food was not up to my expectations. It was just average."
         ]

for text in texts:
    result = pipeline(text)
    print(f"Text: {text}\nSentiment: {result}\n")


