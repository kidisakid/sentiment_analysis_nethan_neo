"""
Merge Page

Main entry point for the Streamlit UI application.
Launches the interactive sentiment analysis interface.
"""

from src.sentiment import run_streamlit_ui


def merge():
    """Launch the Streamlit UI application."""
    run_streamlit_ui()
