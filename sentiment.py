# Using bert for sentiment analysis
# https://huggingface.co/transformers/v3.0.2/model_doc/

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pipeline = pipeline(
    task='sentiment-analysis',
    # Load the model and tokenizer from Hugging Face
    model='distilbert-base-uncased-finetuned-sst-2-english',
    max_length=512,
)
