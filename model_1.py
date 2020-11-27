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




hidden_layers = [
    Conv1D(filters = 300, kernel_size = 19,  activation='relu'),
    BatchNormalization(), 
    MaxPooling1D(pool_size=3),
    Conv1D(filters = 200, kernel_size = 11,  activation='relu'),
    BatchNormalization(), 
    MaxPooling1D(pool_size=4),
    Conv1D(filters = 200, kernel_size = 7,  activation='relu'),
    BatchNormalization(), 
    MaxPooling1D(pool_size=4),
    Dense(1000, activation='relu'),
    Dropout(0.3),
    Dense(1000, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
]
