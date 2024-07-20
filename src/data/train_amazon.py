import time
import re

import pandas as pd
import numpy as np
import nltk
import keras
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from tqdm import tqdm
from sklearn.metrics import *
from scikitplot.metrics import plot_confusion_matrix
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras import layers
from keras import models

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

dataset = pd.read_csv(r".\data\IMDB.csv", sep='\t')
dataset['Reviews'] = dataset['Reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
dataset['Reviews'] = dataset['Reviews'].apply(lambda x: deEmojify(x))
dataset['Reviews'] = dataset['Reviews'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


dataset = pd.read_csv(r".\data\amazon_electronics_review.csv", sep='\t', index_col=[0])
dataset.sample(frac=1).head(5) # shuffle the df and pick first 5
dataset = dataset.iloc[:34000]
dataset['Label'] = dataset['overall'].apply(lambda x: 'negative' if x<=3 else 'positive')

train, test = train_test_split(dataset, test_size=0.2, random_state=200)

nn_label = train['Label'].map({'positive': 1, 'negative':0})
nn_label = tf.cast(nn_label, tf.float32)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Create feature vectors
# TFIDF vectorizer
vectorizer = TfidfVectorizer(min_df = 0.0,
                             max_df = 0.9,
                             sublinear_tf = True,
                             use_idf = True)
arr = []
def chunkIterator(df):
    for data_point in df['reviewText'].values.astype('U'):
        yield data_point
#arr = vectorizer.fit_transform(chunkIterator(train))
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
report = classification_report(test['Label'], prediction_linear, digits=4)

#rcParams['figure.figsize'] = 10,5
plot_confusion_matrix(prediction_linear,test['Label'])
acc_score = accuracy_score(prediction_linear,test['Label'])
pre_score = precision_score(prediction_linear,test['Label'], average='macro')
rec_score = recall_score(prediction_linear,test['Label'],average='macro')
f1score = f1_score(prediction_linear,test['Label'],average='macro')
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print('F1 score', f1score)
print("-"*50)
cr = classification_report(test['Label'], prediction_linear, digits=4)
print(cr)

model = models.Sequential()
model.add(layers.Dense(128, kernel_initializer ='glorot_uniform',input_dim=train_vectors.shape[1]))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(128, kernel_initializer ='glorot_uniform'))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(units= 1, kernel_initializer ='glorot_uniform', activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['acc',f1_m,precision_m, recall_m])

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, 
                                   verbose=0, mode='min', start_from_epoch=3, restore_best_weights=True)

model.summary()

model.fit(train_vectors, nn_label, batch_size = 16, epochs= 100,callbacks=[es],validation_split=0.2, verbose=2)
pred_nn = model.predict(test_vectors)
pred_nn = np.rint(pred_nn).tolist()
pred_nn = [
    x
    for xs in pred_nn
    for x in xs
]
pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
print("Neural network ...")
cr = classification_report(test['Label'], pred_nn, digits=4)
print(cr)

# Count vectorizer
vectorizer = CountVectorizer(min_df = 0.0,
                             max_df = 0.9)
arr = []
def chunkIterator(df):
    for data_point in df['reviewText'].values.astype('U'):
        yield data_point
#arr = vectorizer.fit_transform(chunkIterator(train))
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
report = classification_report(test['Label'], prediction_linear, digits=4)

#rcParams['figure.figsize'] = 10,5
plot_confusion_matrix(prediction_linear,test['Label'])
acc_score = accuracy_score(prediction_linear,test['Label'])
pre_score = precision_score(prediction_linear,test['Label'], average='macro')
rec_score = recall_score(prediction_linear,test['Label'],average='macro')
f1score = f1_score(prediction_linear,test['Label'],average='macro')
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print('F1 score', f1score)
print("-"*50)
cr = classification_report(test['Label'], prediction_linear, digits=4)
print(cr)

model = models.Sequential()
model.add(layers.Dense(128, kernel_initializer ='glorot_uniform',input_dim=train_vectors.shape[1]))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(128, kernel_initializer ='glorot_uniform'))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(units= 1, kernel_initializer ='glorot_uniform', activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['acc',f1_m,precision_m, recall_m])

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, 
                                   verbose=0, mode='min', start_from_epoch=3, restore_best_weights=True)

model.summary()

model.fit(train_vectors, nn_label, batch_size = 16, epochs= 100,callbacks=[es],validation_split=0.2, verbose=2)
pred_nn = model.predict(test_vectors)
pred_nn = np.rint(pred_nn).tolist()
pred_nn = [
    x
    for xs in pred_nn
    for x in xs
]
pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
print("Neural network ...")
cr = classification_report(test['Label'], pred_nn, digits=4)
print(cr)

# Hashing vectorizer
vectorizer = HashingVectorizer()
arr = []
def chunkIterator(df):
    for data_point in df['reviewText'].values.astype('U'):
        yield data_point
#arr = vectorizer.fit_transform(chunkIterator(train))
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
report = classification_report(test['Label'], prediction_linear, digits=4)

#rcParams['figure.figsize'] = 10,5
plot_confusion_matrix(prediction_linear,test['Label'])
acc_score = accuracy_score(prediction_linear,test['Label'])
pre_score = precision_score(prediction_linear,test['Label'], average='macro')
rec_score = recall_score(prediction_linear,test['Label'],average='macro')
f1score = f1_score(prediction_linear,test['Label'],average='macro')
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print('F1 score', f1score)
print("-"*50)
cr = classification_report(test['Label'], prediction_linear, digits=4)
print(cr)

model = models.Sequential()
model.add(layers.Dense(128, kernel_initializer ='glorot_uniform',input_dim=train_vectors.shape[1]))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(128, kernel_initializer ='glorot_uniform'))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(units= 1, kernel_initializer ='glorot_uniform', activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['acc',f1_m,precision_m, recall_m])

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, 
                                   verbose=0, mode='min', start_from_epoch=3, restore_best_weights=True)

model.summary()

model.fit(train_vectors, nn_label, batch_size = 16, epochs= 100,callbacks=[es],validation_split=0.2, verbose=2)
pred_nn = model.predict(test_vectors)
pred_nn = np.rint(pred_nn).tolist()
pred_nn = [
    x
    for xs in pred_nn
    for x in xs
]
pred_nn = pd.Series(pred_nn).map({1.0: 'positive', 0.0:'negative'})
print("Neural network ...")
cr = classification_report(test['Label'], pred_nn, digits=4)
print(cr)