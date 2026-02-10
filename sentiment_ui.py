import streamlit as st
from sentiment import pipeline
import pandas as pd

st.header("Sentiment Analysis")

uploaded_file = st.file_uploader("Upload csv file for sentiment analysis", type=["csv"])
df = pd.read_csv(uploaded_file) if uploaded_file is not None else None

column_list = []
if df is not None:
    for columns in df.columns:
        column_list.append(columns)
else:
    st.write("Please upload a csv file for sentiment analysis")

content = st.selectbox("Select the content column", column_list)

if st.button("Analyze"):
    analysis = []
    
    with st.spinner('Analyzing...'):
        for text in df[content]:
            result = pipeline(text)

            if result[0]['label'] == 'LABEL_0':
                result[0]['label'] = 'Negative'
            elif result[0]['label'] == 'LABEL_1':
                result[0]['label'] = 'Neutral'
            elif result[0]['label'] == 'LABEL_2':
                result[0]['label'] = 'Positive'

            analysis.append(result[0]['label'].capitalize())

    #Add sentiment column
    df['Sentiment'] = analysis 
    st.write(df[[content, 'Sentiment']])
    
    #Download button
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name=uploaded_file.name.split('.')[0] + '_sentiment.csv',
        mime='text/csv',
    )