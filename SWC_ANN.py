import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import layers, models, optimizers
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import adam
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sn


train = pd.read_csv("D:/Python/data/SWC/traindata.csv")

pd.options.display.max_columns = train.shape[1]

train_values = train[train.columns.difference(['target'])]
train_target = train[['target']] - 1

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0, stratify = train_target)


##y_test_cm = y_test

y_train.target.value_counts()
y_test.target.value_counts()

X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
X_test = preprocessing.MinMaxScaler().fit_transform(X_test)

y_train = np_utils.to_categorical(y_train.values.ravel())
y_test = np_utils.to_categorical(y_test.values.ravel())



dnn_model_1 = tf.keras.models.Sequential()

dnn_model_1.add(tf.keras.layers.Dense(units = 412, input_dim = 103, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 412, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 412, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 412, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 412, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 206, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dense(units = 103, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dense(units = 9, activation = tf.nn.softmax))


dnn_model_1.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
dnn_model_1.summary()

dnn_model_1.fit(X_train, y_train, epochs = 50)


dnn_model_1_prob = dnn_model_1.predict(X_test)

dnn_model_1_log_loss = log_loss(y_test, dnn_model_1_prob)
print('dnn_model_1_log_loss: ', dnn_model_1_log_loss)


cm_dnn_model_1 = confusion_matrix(y_test_cm, np.asarray([np.argmax(line) for line in dnn_model_1_prob]))

plt.figure(figsize = (15, 10))
sn.heatmap(cm_dnn_model_1, annot = True, fmt='.5g').set(xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                                        yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.xlabel('Predict')
plt.ylabel('Actual')

























