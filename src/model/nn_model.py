import keras
import tensorflow as tf
import tensorflow.keras.backend as K

from keras import layers
from keras import models

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

def nn_model(train, labels):
    model = models.Sequential()
    model.add(layers.Dense(128, kernel_initializer ='glorot_uniform',input_dim=train.shape[1]))
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
    model.fit(train, labels, batch_size = 16, epochs= 100,callbacks=[es],validation_split=0.2, verbose=0)
    return model