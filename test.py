import pandas as pd
import ast

def get_sentiments(row):
    text = row['Sentiment Object']
    try:
        sent_dict = ast.literal_eval(text)
        sentiment = sent_dict['documents'][0]['document_sentiment']
        confidence = sent_dict['documents'][0]['document_scores'][f'{sentiment}']
        return sentiment, confidence
    except:
        return 'Error', 1

data = pd.read_csv('.\data\service_tickets4.csv', sep='\t', index_col=[0])
data[['sentiment', 'confidence']] = data.apply(get_sentiments, axis=1, result_type='expand')
data.to_csv('.\data\Cherwell.csv', index=False, sep='\t')
pass