import time

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

dataset = pd.read_csv(r".\data\amazon_electronics_review.csv", sep='\t', index_col=[0])
dataset.sample(frac=1).head(5) # shuffle the df and pick first 5
dataset = dataset.iloc[:34000]
dataset['Label'] = dataset['overall'].apply(lambda x: 'negative' if x<3 else ('neutral' if x==3 else 'positive'))

train, test = train_test_split(dataset, test_size=0.2)
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 0.0,
                             max_df = 0.9,
                             sublinear_tf = True,
                             use_idf = True)
arr = []
def chunkIterator(df):
    for data_point in df['reviewText'].values.astype('U'):
        yield data_point
arr = vectorizer.fit_transform(chunkIterator(train))
train_vectors = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
test_vectors = vectorizer.transform(test['reviewText'].values.astype('U'))

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train['Label'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(test['Label'], prediction_linear, output_dict=True)

review = """SUPERB, I AM IN LOVE IN THIS PHONE"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """Do not purchase this product. My cell phone blast when I switched the charger"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """This product doesn't work correctly"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

