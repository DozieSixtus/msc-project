from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn import svm

# Example text data
dataset = pd.read_csv(r".\data\amazon_electronics_review.csv", sep='\t', index_col=[0])
dataset.sample(frac=1).head(5) # shuffle the df and pick first 5
dataset = dataset.iloc[:34000]
dataset['Label'] = dataset['overall'].apply(lambda x: 'negative' if x<=3 else 'positive')
dataset = dataset.fillna('Null')

train, test = train_test_split(dataset, test_size=0.2, random_state=200)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data to get TF-IDF vectors
tfidf_vectors = tfidf_vectorizer.fit_transform(train['reviewText']).toarray()

from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModel
import torch

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

# Function to get embeddings from BERT
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
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
classifier = svm.LinearSVC()
classifier.fit(combined_features, labels)

# Example of making predictions
predictions = classifier.predict(combined_features)
print(predictions)

test_vectors = tfidf_vectorizer.transform(test['reviewText']).toarray()
test_bert_embeddings = get_bert_embeddings(test['reviewText'].to_list())
combined_test_vectors = np.hstack((test_vectors, test_bert_embeddings))

predictions = classifier.predict(combined_test_vectors)
cr = classification_report(test['Label'], predictions, digits=4)
print(cr)
