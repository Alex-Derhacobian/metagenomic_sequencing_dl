import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Bio import SeqIO
import gzip
import math
from google.colab import drive
from os import listdir
from os.path import join, exists
import re
if not exists('/content/drive'):
    drive.mount('/content/drive')
import random
import h5py
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

# set seed to be the same
random.seed(0)

#keras imports and such
# Some basic model things
%tensorflow_version 2.x
import tensorflow.python.keras
import tensorflow as tf
# from tensorflow.python.keras
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D, BatchNormalization, MaxPooling1D
from tensorflow.python.keras.layers.merge import Average
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
from keras import regularizers


# parameters
POOL_FACTOR = 1
dropout_cnn = 0.1
dropout_pool = 0.1
dropout_dense = 0.1
learningrates = 10**(-3-1*np.random.rand(7))

channel_num  = 4
contigLength = 500
filter_len1= 10
nb_filter1 = 1000
nb_dense = 1000

batch_size=32
pool_len1 = int((contigLength-filter_len1+1)/POOL_FACTOR)
beta_1=0.9
beta_2=0.999

print(batch_size)
print(pool_len1)
print(learningrates)

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

forward_input = Input(shape=(w, channel_num))
reverse_input = Input(shape=(w, channel_num))
hidden_layers = [
    Conv1D(filters = nb_filter1, kernel_size = filter_len1, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    GlobalMaxPooling1D(),
    Dropout(dropout_pool),
    Dense(nb_dense, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(dropout_dense),
    Dense(1, activation='sigmoid')
]
forward_output = get_output(forward_input, hidden_layers)     
reverse_output = get_output(reverse_input, hidden_layers)
output = Average()([forward_output, reverse_output])
model = Model(inputs=[forward_input, reverse_input], outputs=output)
print(model.summary())

history = []
for i in range(len(learningrates)):
    print("Testing with learning rate = " + str(learningrates[i]))
    model_dir = '/content/drive/My Drive/cs230_metagenomics/BugNet/bas_test/saved_models'
    checkpointer = ModelCheckpoint(filepath=join(model_dir, 'lr_' + str(i) +'-{epoch:02d}-{val_accuracy:.2f}.hdf5'),
                                     verbose=1,save_best_only=True, 
                                      monitor='val_accuracy')

    earlystopper = EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=5, verbose=1)
    opt = tf.keras.optimizers.Adam(lr=learningrates[i], beta_1=beta_1, beta_2=beta_2, amsgrad=False)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    train = True
    if train:

        history.append(model.fit(x=[X_train_forw, X_train_rev], y=Y_train, 
            batch_size=batch_size, epochs=10, verbose=2,
            validation_data=([X_dev_forw, X_dev_rev], Y_dev),
            callbacks=[checkpointer, earlystopper]))
"""
else: 
    # load in predetermined weights
    listdir(model_dir)
    model.load_weights(join(model_dir, "weights-11-0.88.hdf5"))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
"""

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()

