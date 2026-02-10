from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pipeline = pipeline(
    task='sentiment-analysis',
    # Load the model and tokenizer from Hugging Face
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    max_length=512,
)
