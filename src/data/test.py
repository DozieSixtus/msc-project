from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Example text data
dataset = pd.read_csv(r".\data\amazon_electronics_review.csv", sep='\t', index_col=[0])
dataset.sample(frac=1).head(5) # shuffle the df and pick first 5
dataset = dataset.iloc[:500]
dataset['Label'] = dataset['overall'].apply(lambda x: 'negative' if x<=3 else 'positive')
dataset = dataset.fillna('Null')

train, test = train_test_split(dataset, test_size=0.2, random_state=200)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data to get TF-IDF vectors
tfidf_vectors = tfidf_vectorizer.fit_transform(train['reviewText']).toarray()

from transformers import AlbertTokenizer, AlbertModel
import torch

# Initialize the tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')

# Function to get embeddings from BERT
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings from the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Get BERT embeddings for the text data
bert_embeddings = get_bert_embeddings(train['reviewText'].to_list())
import numpy as np

# Concatenate TF-IDF vectors with BERT embeddings
combined_features = np.hstack((tfidf_vectors, bert_embeddings))

from sklearn.ensemble import RandomForestClassifier

# Example labels for the text data
labels = train['Label']

# Train a classifier using the combined feature vectors
classifier = RandomForestClassifier(verbose=1)
classifier.fit(combined_features, labels)

# Example of making predictions
predictions = classifier.predict(combined_features)
print(predictions)
