"""
Sentiment Analysis Application

Simple entry point that launches the interactive Streamlit UI.
"""

import subprocess
import sys
from src.sentiment_ui import sentiment

if __name__ == "__main__":
    sentiment()