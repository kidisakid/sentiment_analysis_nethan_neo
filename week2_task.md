Goal:
Classify text as Positive / Neutral / Negative using BERT
Provide a simple Streamlit UI within app.py for testing

Tasks / Requirements:
1. Create ML/sentiment.py with a predict_sentiment(text) function:
- Tokenizes input
- Runs it through BERT
- Returns Positive / Neutral / Negative
- Test the module with sample text and verify outputs

2. Create a new branch (e.g., machine_learning/sentiment-analysis) and open a PR:
- Include purpose, dependencies, and example usage in PR description
- Update app.py to include Streamlit UI:
- Let the user select the column to do sentiment analysis
- Button/selection to run that specific module

3. Display result (Positive / Neutral / Negative)
- Update existing README.md:
- Document ML module usage
- Document Streamlit UI usage in app.py
- Reflect updated dependencies
