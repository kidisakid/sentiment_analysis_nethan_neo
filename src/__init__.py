"""
Sentiment Analysis Package

This package provides sentiment classification utilities using BERT transformer models.
Includes both a machine learning module for programmatic access and a Streamlit UI
for interactive analysis.

Modules:
    sentiment: Core sentiment analysis functionality with BERT pipeline
    sentiment_ui: Interactive Streamlit web interface

Usage:
    from src.sentiment import pipeline
    
    text = "I love this product!"
    result = pipeline(text)
    print(result)
"""

from .model import pipeline

__version__ = "1.0.0"
__all__ = ["pipeline"]
