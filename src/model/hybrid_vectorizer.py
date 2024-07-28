import pandas as pd
import torch
import numpy as np
import tensorflow as tf

from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from nn_model import nn_model


# Example text data
dataset = pd.read_csv(r".\data\amazon_electronics_review.csv", sep='\t', index_col=[0])
dataset.sample(frac=1, random_state=42).head(5) # shuffle the df and pick first 5
dataset = dataset[~dataset['reviewText'].isna()]
dataset['Label'] = dataset['overall'].apply(lambda x: 'negative' if x<3 else 'positive' if x>3 else 'neutral')
dataset = pd.concat([dataset[dataset['Label']=='negative'][:17000],
                    dataset[dataset['Label']=='positive'][:17000]], ignore_index=True)

#dataset = dataset.fillna('Null')

train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data to get TF-IDF vectors
tfidf_vectors = tfidf_vectorizer.fit_transform(train['reviewText']).toarray()

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

# Concatenate TF-IDF vectors with BERT embeddings
combined_features = np.hstack((tfidf_vectors, bert_embeddings))

# Example labels for the text data
labels = train['Label']

# Train a classifier using the combined feature vectors
classifier = svm.LinearSVC()
classifier.fit(combined_features, labels)
labels = train['Label'].map({'positive': 1, 'negative':0})
labels = tf.cast(labels, tf.float32)
classifier = nn_model(combined_features, labels)

# Example of making predictions
predictions = classifier.predict(combined_features)
print(predictions)

test_vectors = tfidf_vectorizer.transform(test['reviewText']).toarray()
test_bert_embeddings = get_bert_embeddings(test['reviewText'].to_list())
combined_test_vectors = np.hstack((test_vectors, test_bert_embeddings))

predictions = classifier.predict(combined_test_vectors)
test['Label'] = test['Label'].map({'positive': 1, 'negative':0})
cr = classification_report(test['Label'], predictions, digits=4)
print(cr)

