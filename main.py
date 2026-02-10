"""
Sentiment Analysis Application

Simple entry point that launches the interactive Streamlit UI.
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/sentiment_ui.py"],
        check=True
    )
