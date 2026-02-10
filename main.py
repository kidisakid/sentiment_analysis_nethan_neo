"""
Sentiment Analysis Orchestrator

Main entry point for the sentiment analysis application.
Provides multiple ways to use the sentiment analyzer:
1. Interactive Streamlit UI for CSV analysis
2. Command-line batch processing
3. Direct Python API access

Usage:
    # Run Streamlit UI (interactive)
    python main.py --ui
    
    # Analyze a single text
    python main.py --text "I love this product!"
    
    # Analyze CSV file
    python main.py --csv data/reviews.csv --column review_text
    
    # Show help
    python main.py --help
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from src.sentiment import pipeline


def analyze_text(text: str) -> dict:
    """
    Analyze sentiment of a single text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with label and score
    """
    result = pipeline(text)
    return result[0]


def analyze_csv(csv_path: str, column: str, output_path: str = None) -> pd.DataFrame:
    """
    Analyze sentiment for all texts in a CSV column.
    
    Args:
        csv_path: Path to input CSV file
        column: Column name containing text to analyze
        output_path: Optional path to save results. Defaults to 
                    {input_filename}_sentiment.csv
        
    Returns:
        DataFrame with original data plus Sentiment column
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If specified column doesn't exist in CSV
    """
    # Validate file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    print(f"📂 Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Validate column exists
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )
    
    # Analyze sentiment for each row
    print(f"🔄 Analyzing sentiment for {len(df)} rows...")
    sentiments = []
    
    for idx, text in enumerate(df[column], 1):
        result = pipeline(str(text))
        label = result[0]['label']
        
        # Map BERT labels to readable format
        if label == 'LABEL_0':
            sentiment = 'Negative'
        elif label == 'LABEL_1':
            sentiment = 'Neutral'
        elif label == 'LABEL_2':
            sentiment = 'Positive'
        else:
            sentiment = label
        
        sentiments.append(sentiment)
        
        # Progress indicator
        if idx % max(1, len(df) // 10) == 0:
            print(f"   {idx}/{len(df)} rows processed...")
    
    df['Sentiment'] = sentiments
    
    # Save results
    if output_path is None:
        input_name = Path(csv_path).stem
        output_path = f"{input_name}_sentiment.csv"
    
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")
    
    return df


def run_streamlit_ui():
    """Launch the Streamlit UI application."""
    import subprocess
    
    print("🚀 Launching Streamlit UI...")
    print("   Open your browser to http://localhost:8501")
    
    subprocess.run(
        ["streamlit", "run", "src/sentiment_ui.py"],
        check=True
    )


def main():
    """Parse arguments and execute appropriate action."""
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Tool - Classify text as Positive/Neutral/Negative",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive UI
  python main.py --ui
  
  # Analyze single text
  python main.py --text "This product is amazing!"
  
  # Analyze CSV file (saves to reviews_sentiment.csv)
  python main.py --csv data/reviews.csv --column review_text
  
  # Analyze CSV with custom output path
  python main.py --csv data/reviews.csv --column review_text --output results.csv
        """
    )
    
    # Create mutually exclusive group for action
    action = parser.add_mutually_exclusive_group(required=False)
    action.add_argument(
        '--ui',
        action='store_true',
        help='Launch interactive Streamlit UI'
    )
    action.add_argument(
        '--text',
        type=str,
        help='Analyze sentiment of a single text'
    )
    action.add_argument(
        '--csv',
        type=str,
        help='Analyze sentiment for CSV file'
    )
    
    # Additional arguments
    parser.add_argument(
        '--column',
        type=str,
        help='Column name in CSV containing text (required with --csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for analyzed CSV (optional with --csv)'
    )
    
    args = parser.parse_args()
    
    # Default to UI if no arguments provided
    if len(sys.argv) == 1:
        args.ui = True
    
    try:
        if args.ui or (not args.text and not args.csv):
            # Launch Streamlit UI
            run_streamlit_ui()
            
        elif args.text:
            # Analyze single text
            print(f"📝 Analyzing: {args.text}\n")
            result = analyze_text(args.text)
            
            # Map label to readable format
            label = result['label']
            if label == 'LABEL_0':
                sentiment = 'Negative'
            elif label == 'LABEL_1':
                sentiment = 'Neutral'
            elif label == 'LABEL_2':
                sentiment = 'Positive'
            else:
                sentiment = label
            
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {result['score']:.2%}")
            
        elif args.csv:
            # Analyze CSV file
            if not args.column:
                parser.error("--column is required when using --csv")
            
            analyze_csv(args.csv, args.column, args.output)
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
