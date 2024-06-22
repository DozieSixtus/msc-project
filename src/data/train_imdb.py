import time

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from tqdm import tqdm
from sklearn.metrics import *
from scikitplot.metrics import plot_confusion_matrix

dataset = pd.read_csv(r".\data\IMDB.csv", sep='\t')

train, test = train_test_split(dataset, test_size=0.2, random_state=200)

# Create feature vectors
# TFIDF vectorizer
vectorizer = TfidfVectorizer(min_df = 0.0,
                             max_df = 0.9,
                             sublinear_tf = True,
                             use_idf = True)
arr = []
def chunkIterator(df):
    for data_point in df['Reviews'].values.astype('U'):
        yield data_point
#arr = vectorizer.fit_transform(chunkIterator(train))
train_vectors = vectorizer.fit_transform(train['Reviews'].values.astype('U'))
test_vectors = vectorizer.transform(test['Reviews'].values.astype('U'))

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
report = classification_report(test['Label'], prediction_linear)

review = """SUPERB, I AM IN LOVE IN THIS PHONE"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """Do not purchase this product. My cell phone blast when I switched the charger"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """This product doesn't work correctly"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

#rcParams['figure.figsize'] = 10,5
plot_confusion_matrix(prediction_linear,test['Label'])
acc_score = accuracy_score(prediction_linear,test['Label'])
pre_score = precision_score(prediction_linear,test['Label'], average=None)
rec_score = recall_score(prediction_linear,test['Label'],average=None)
f1score = f1_score(prediction_linear,test['Label'],average=None)
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print('F1 score', f1score)
print("-"*50)
cr = classification_report(test['Label'], prediction_linear)
print(cr)

# Count vectorizer
vectorizer = CountVectorizer(min_df = 0.0,
                             max_df = 0.9)
arr = []
def chunkIterator(df):
    for data_point in df['Reviews'].values.astype('U'):
        yield data_point
#arr = vectorizer.fit_transform(chunkIterator(train))
train_vectors = vectorizer.fit_transform(train['Reviews'].values.astype('U'))
test_vectors = vectorizer.transform(test['Reviews'].values.astype('U'))

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
report = classification_report(test['Label'], prediction_linear)

review = """SUPERB, I AM IN LOVE IN THIS PHONE"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """Do not purchase this product. My cell phone blast when I switched the charger"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """This product doesn't work correctly"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

#rcParams['figure.figsize'] = 10,5
plot_confusion_matrix(prediction_linear,test['Label'])
acc_score = accuracy_score(prediction_linear,test['Label'])
pre_score = precision_score(prediction_linear,test['Label'], average=None)
rec_score = recall_score(prediction_linear,test['Label'],average=None)
f1score = f1_score(prediction_linear,test['Label'],average=None)
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print('F1 score', f1score)
print("-"*50)
cr = classification_report(test['Label'], prediction_linear)
print(cr)

# Hashing vectorizer
vectorizer = HashingVectorizer()
arr = []
def chunkIterator(df):
    for data_point in df['Reviews'].values.astype('U'):
        yield data_point
#arr = vectorizer.fit_transform(chunkIterator(train))
train_vectors = vectorizer.fit_transform(train['Reviews'].values.astype('U'))
test_vectors = vectorizer.transform(test['Reviews'].values.astype('U'))

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
report = classification_report(test['Label'], prediction_linear)

review = """SUPERB, I AM IN LOVE IN THIS PHONE"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """Do not purchase this product. My cell phone blast when I switched the charger"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

review = """This product doesn't work correctly"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

#rcParams['figure.figsize'] = 10,5
plot_confusion_matrix(prediction_linear,test['Label'])
acc_score = accuracy_score(prediction_linear,test['Label'])
pre_score = precision_score(prediction_linear,test['Label'], average=None)
rec_score = recall_score(prediction_linear,test['Label'],average=None)
f1score = f1_score(prediction_linear,test['Label'],average=None)
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print('F1 score', f1score)
print("-"*50)
cr = classification_report(test['Label'], prediction_linear)
print(cr)