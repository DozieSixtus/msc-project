import pandas as pd
import torch
import numpy as np
import tensorflow as tf

from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from nn_model import nn_model

def get_tfidf_vectors(train, test):
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df = 0.0,
                             max_df = 0.9,
                             sublinear_tf = True,
                             use_idf = True)

    # Fit and transform the text data to get TF-IDF vectors
    train_vectors = tfidf_vectorizer.fit_transform(train.values.astype('U'))
    test_vectors = tfidf_vectorizer.transform(test.values.astype('U'))

    return train_vectors, test_vectors


# Function to get embeddings from BERT
def get_bert_embeddings(texts):
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    
    inputs = tokenizer(texts, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings from the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def get_count_vectors(train, test):
    # Count vectorizer
    vectorizer = CountVectorizer(min_df = 0.0,
                                max_df = 0.9)

    train_vectors = vectorizer.fit_transform(train.values.astype('U'))
    test_vectors = vectorizer.transform(test.values.astype('U'))
    
    return train_vectors, test_vectors

def get_hashing_vectors(train, test):
    # Hashing vectorizer
    vectorizer = HashingVectorizer()

    train_vectors = vectorizer.fit_transform(train.values.astype('U'))
    test_vectors = vectorizer.transform(test.values.astype('U'))
    
    return train_vectors, test_vectors