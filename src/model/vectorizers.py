import pandas as pd
import torch
import numpy as np
import nltk

from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
# Path to GloVe embeddings file (ensure you have this file)
glove_file_path = 'glove.6B.300d.txt'

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

def get_glove_vectors(text_df, embedding_dim=300):
    # download glove and unzip it in Notebook.
    #!wget http://nlp.stanford.edu/data/glove.6B.zip
    #!unzip glove*.zip

    # Load pre-trained GloVe embeddings
    def load_glove_embeddings(glove_file_path):
        embeddings = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    # Convert text to GloVe embeddings
    def text_to_glove(text, embeddings, embedding_dim):
        words = word_tokenize(text.lower())
        word_embeddings = [embeddings[word] for word in words if word in embeddings]
        
        if len(word_embeddings) > 0:
            word_embeddings = np.mean(word_embeddings, axis=0)
            return word_embeddings
        else:
            return np.zeros(embedding_dim)

    # Load GloVe embeddings
    glove_embeddings = load_glove_embeddings(glove_file_path)

    # Convert text to GloVe embeddings and prepare data
    text_vectors = np.array([text_to_glove(text, glove_embeddings, embedding_dim) for text in text_df])
    return text_vectors
