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



# params for loading data
n_train_phage = 1000
n_train_bact = 250
n_dev = 100
# phage genomes look like they're on average 2-3% as long as bacterial
# keep 5% and subsample down from that later
keep_bact_frac = 0.05
w = 1000
# load some training data
load_new = False
clear_mem = False
data_file= "/content/drive/My Drive/cs230_metagenomics/BugNet/bas_test/data_w" + str(w) + ".hdf5"
if load_new:
    print('Loading phage training data')
    with h5py.File(data_file, "w") as f:
        train_phage_forw  = load_many_fasta(train_phage_files[0:n_train_phage], w=w, keep_frac=1.0)
        f.create_dataset('train_phage_forw', data=train_phage_forw, compression="gzip")
        if clear_mem:
            del train_phage_forw
        print('Loading bacteria training data')
        train_bact_forw  = load_many_fasta(train_bact_files[0:n_train_bact], w=w, keep_frac=keep_bact_frac)
        f.create_dataset('train_bact_forw', data=train_bact_forw, compression="gzip")
        if clear_mem:
            del train_bact_forw
        # dev set
        print('Loading phage dev data')
        dev_phage_forw  = load_many_fasta(dev_phage_files[0:n_dev], w=w, keep_frac=1.0)
        f.create_dataset('dev_phage_forw', data=dev_phage_forw, compression="gzip")
        if clear_mem:
            del dev_phage_forw
        print('Loading bacteria dev data')
        dev_bact_forw  = load_many_fasta(dev_bact_files[0:n_dev], w=w, keep_frac=keep_bact_frac)
        f.create_dataset('dev_bact_forw', data=dev_bact_forw, compression="gzip")
        if clear_mem:
            del dev_bact_forw

    if clear_mem:
        sys.exit()

else:
    print('loading data')
    with  h5py.File(data_file, "r") as f:
        print(' ... training phage')
        train_phage_forw = np.array(f.get('train_phage_forw'))
        print(' ... training bact')
        train_bact_forw = np.array(f.get('train_bact_forw'))
        print(' ... dev phage')
        dev_phage_forw = np.array(f.get('dev_phage_forw'))
        print(' ... dev bact')
        dev_bact_forw = np.array(f.get('dev_bact_forw'))
    print('finished loading')

print('making reverse complements')
train_phage_rev = rev_comp_many(train_phage_forw)
train_bact_rev = rev_comp_many(train_bact_forw)
dev_phage_rev = rev_comp_many(dev_phage_forw)
dev_bact_rev = rev_comp_many(dev_bact_forw)
print(train_phage_forw.shape)
print(train_bact_forw.shape)
print(train_phage_forw.shape[0] / train_bact_forw.shape[0])
print(dev_phage_forw.shape[0] / dev_bact_forw.shape[0])

train_imbalance = train_phage_forw.shape[0] / train_bact_forw.shape[0]
dev_imbalance = dev_phage_forw.shape[0] / dev_bact_forw.shape[0]
# still need to subsample the bacteria data, even after that correction
if train_imbalance < 0.99:
    keep_mask = np.random.rand(train_bact_forw.shape[0]) < train_imbalance
    train_bact_forw = train_bact_forw[keep_mask]
    train_bact_rev = rev_comp_many(train_bact_forw)
if dev_imbalance < 0.99:
    keep_mask = np.random.rand(dev_bact_forw.shape[0]) < dev_imbalance
    dev_bact_forw = dev_bact_forw[keep_mask]
    dev_bact_rev = rev_comp_many(dev_bact_forw)

print(train_phage_forw.shape[0] / train_bact_forw.shape[0])
print(dev_phage_forw.shape[0] / dev_bact_forw.shape[0])

