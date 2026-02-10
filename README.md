# Sentiment Analysis with BERT

A machine learning-powered sentiment classification tool that analyzes text data and classifies emotions as **Positive**, **Neutral**, or **Negative** using state-of-the-art transformer models.

## Overview

This project provides two ways to perform sentiment analysis:
1. **Python Module** (`sentiment.py`) - Programmatic access for batch processing
2. **Streamlit UI** (`sentiment_ui.py`) - Interactive web interface for user-friendly analysis

## Dependencies

- **transformers** (4.0+) - Hugging Face library for pre-trained NLP models
- **streamlit** (1.0+) - Framework for building interactive web apps
- **pandas** - Data manipulation and CSV handling

Install dependencies using:
```bash
pip install -r requirements.txt
```

## ML Module Usage (`sentiment.py`)

### Purpose
The sentiment analysis module uses a pre-trained BERT-based model (`cardiffnlp/twitter-roberta-base-sentiment`) to classify text sentiment with high accuracy.

### How It Works
- Tokenizes input text (max 512 characters)
- Runs the tokenized text through the transformer model
- Returns sentiment labels: `LABEL_0` (Negative), `LABEL_1` (Neutral), `LABEL_2` (Positive)

### Sample Usage

```python
from src.sentiment import pipeline

# Single text analysis
text = "I absolutely love this product!"
result = pipeline(text)
print(result)  
# Output: [{'label': 'LABEL_2', 'score': 0.99}]

# Process multiple texts
texts = [
    "This is amazing!",
    "It's okay, nothing special",
    "Terrible experience"
]

for text in texts:
    result = pipeline(text)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
```

### Model Details
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Input Limit**: 512 tokens
- **Output Format**: List of dictionaries with `label` and `score` keys
- **Labels**:
  - `LABEL_0` = Negative
  - `LABEL_1` = Neutral
  - `LABEL_2` = Positive

## Streamlit UI Usage (`sentiment_ui.py`)

### Purpose
The Streamlit UI provides an interactive web interface for analyzing sentiment in CSV files. Users can upload data, select columns for analysis, and download results with sentiment labels.

### Features
✨ **User-Friendly Interface**
- Upload CSV files directly
- Select which column to analyze
- Real-time sentiment analysis with progress indicator
- View results in formatted table
- Download analyzed CSV with sentiment column

### How to Run

1. **Start the Streamlit app**:
   ```bash
   streamlit run src/sentiment_ui.py
   ```

2. **Open in browser**: Navigate to `http://localhost:8501`

### Step-by-Step Guide

1. **Upload CSV File**
   - Click "Upload csv file for sentiment analysis"
   - Select your CSV file from your computer

2. **Select Column**
   - Choose the column containing text to analyze from the dropdown

3. **Analyze**
   - Click the "Analyze" button
   - Wait for analysis to complete (spinner indicates progress)

4. **View Results**
   - Table displays original text and sentiment classification

5. **Download**
   - Click "Download CSV" to save results
   - File saved as `{original_filename}_sentiment.csv`

### Example Workflow

**Input CSV** (`reviews.csv`):
```
Review
"Absolutely fantastic product, highly recommend!"
"Average quality, could be better"
"Worst purchase ever, very disappointed"
```

**After Analysis** (`reviews_sentiment.csv`):
```
Review,Sentiment
"Absolutely fantastic product, highly recommend!",Positive
"Average quality, could be better",Neutral
"Worst purchase ever, very disappointed",Negative
```

## Project Structure

```
.
├── main.py                 # Entry point - launches UI
├── ui/
│   ├── __init__.py        # UI package initialization
│   └── merge_page.py      # Main UI launcher
├── src/
│   ├── __init__.py        # ML package initialization
│   ├── sentiment.py        # ML module with all functions
│   └── sentiment_ui.py     # Streamlit UI application
├── data/
│   └── Test_File-2.csv     # Sample test data
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Architecture

```
main.py (Simple entry point)
    ↓
ui/merge_page.py (UI launcher)
    ↓
src/sentiment.py (All core functions)
    ├── analyze_text() - Single text analysis
    ├── analyze_csv() - Batch CSV processing
    ├── run_streamlit_ui() - Launch Streamlit app
    └── main() - CLI interface
    ↓
BERT Model (cardiffnlp/twitter-roberta-base-sentiment)
```

## Getting Started

### Prerequisites
- Python 3.7+
- pip or conda

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sevengen-internshipweek2.1

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**Option 1: Launch Interactive UI** (Recommended)
```bash
python main.py
```
Opens the Streamlit interface at `http://localhost:8501`

**Option 2: CLI Commands** (Using src/sentiment.py directly)
```bash
# Analyze single text
python src/sentiment.py --text "I absolutely love this!"

# Analyze CSV file
python src/sentiment.py --csv data/reviews.csv --column review_text

# Analyze CSV with custom output
python src/sentiment.py --csv data/reviews.csv --column review_text --output results.csv

# Show help
python src/sentiment.py --help
```

**Option 3: Use the Python Module** (For developers/integration)
```python
from src.sentiment import pipeline, analyze_text, analyze_csv

# Single text analysis
text = "I love this product!"
result = analyze_text(text)

# Batch CSV processing
df = analyze_csv('data/reviews.csv', 'review_text')
```

## Main Application (`main.py`)

Simple entry point that launches the interactive Streamlit UI.

```python
from ui.merge_page import merge

if __name__ == "__main__":
    merge()
```

## Core Module (`src/sentiment.py`)

Contains all sentiment analysis functionality.

### Available Functions

| Function | Purpose |
|----------|---------|
| `analyze_text(text)` | Analyze sentiment of single text |
| `analyze_csv(csv_path, column, output_path)` | Batch process CSV file |
| `run_streamlit_ui()` | Launch interactive Streamlit app |
| `main()` | CLI interface with argument parsing |
| `pipeline(text)` | Direct BERT sentiment analysis |
