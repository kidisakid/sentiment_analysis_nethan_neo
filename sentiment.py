# Using bert for sentiment analysis
# https://huggingface.co/transformers/v3.0.2/model_doc/

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pipeline = pipeline(
    task='sentiment-analysis',
    # Load the model and tokenizer from Hugging Face
    model='distilbert-base-uncased-finetuned-sst-2-english',

)

texts = [
    "I love this product! It works great and exceeded my expectations.",
    "This is the worst experience I've ever had. I'm very disappointed.",
    "The movie was okay, not the best but not the worst either.",
    "I am extremely happy with the service. The staff was friendly and helpful.",
    "The food was terrible. I will never eat here again."
    "Good staff, but the food was not up to my expectations. It was just average."
    "Bad experience. The product broke after one use and customer service was unhelpful."
    "I am unsatisfied with the quality of the product. It did not meet my needs and I will be returning it.",
    "I did not enjoy the movie. The plot was predictable and the acting was subpar. I would not recommend it to others."
    ]

for text in texts:
    result = pipeline(text)
    # Print the result magnitude/score as negative, neutral, or positive
    # using score thresholds: negative (<0.4), neutral (0.4-0.6), positive (>0.6)

    print(f"Text: {text}\nSentiment: {result}\n")

