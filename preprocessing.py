import numpy as np
import os
from os import listdir
from os.path import join, exists




def _open(filename, mode='rt'):
    """
    helper function for opening files that allows for gzipped
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def one_hot_arr(arr):
    """
    returns a one hot encoding of an array of dna strings
    """
    enc = OneHotEncoder(categories=[['A','C','G','T']], sparse=False)
    return(np.array([one_hot_string(s) for s in arr]))

def one_hot_string(s, enc=None):
    """
    returns a one hot encoding of a dna string, s
    given an encoded, enc
    """
    assert(type(s) in [str, np.str_])
    if enc is None:
        enc = OneHotEncoder(categories=[['A','C','G','T']], sparse=False)
    # transform s to the correct shape
    s_arr = np.array(list(s)).reshape(-1,1)
    s_enc = enc.fit_transform(s_arr)
    return(s_enc)

def load_fasta(f):
    """
    returns an array of strings from loading a fasta file in f
    file can be gzipped
    For now, removes any non-ATCG characers from the sequence
    """
    seq_arr = []
    fasta = _open(f)
    for record in SeqIO.parse(fasta, 'fasta'):
        seq_arr.append(re.sub('[^ATCG]','',str(record.seq).upper()))
    return(np.array(seq_arr))

def shred_fasta(seq_arr, w, keep_frac=1.0):
    """
    shreds a fasta sequence into non-overlapping windows of size w
    seq_arr: aray of strings representing a fasta file
    optionally keep only a certain fraction of them to downsample bac
    w: window size 
    keep_frac: float, keep this proportion of windows 
    """
    shred_arr = []
    for seq in seq_arr:
        max_w = math.floor(len(seq)/ w)
        for j in range(max_w):
            seq_w=seq[(j*w):((j+1)*w)]
            shred_arr.append(seq_w)
    shred_arr = np.array(shred_arr)
    # subsample the windows
    if (keep_frac < 1.0):
        keep_mask = np.random.rand(shred_arr.shape[0]) < keep_frac
        shred_arr = shred_arr[keep_mask]
    return(shred_arr)

def rev_comp_one(arr):
    """ 
    Returns the reverse complement of a one-hot encoded dna sequence
    """
    assert(type(arr)==np.ndarray)
    assert(len(arr.shape)==2)
    assert(arr.shape[1]==4)
    arr_rev = arr[::-1,:]
    # switch cols: A-T, C-G (0-3, 2-1)
    arr_rev_comp = arr_rev[:,(3,2,1,0)]
    return(arr_rev_comp)

def rev_comp_many(many_arr):
    """
    wrapper for rev_comp_one on many sequence arrays
    many_arr: array of many sequences. shape=(seqs, 4, window_size)
    returns reverse complement of each sequence
    """
    # print(type(many_arr))
    assert(type(many_arr)==np.ndarray)
    assert(len(many_arr.shape) ==3)
    return(np.array([rev_comp_one(x) for x in many_arr]))

def load_many_fasta(f_list, w=500, keep_frac=1.0):
    """
    load and encode each fasta file in f_list
    Shreds into non-overlapping windows of size w
    optionally keep only a certain fraction of windows
    returns: forw_encoded, the one-hot encoded, shredded version of the sequences 
    """
    i = 1
    forw_encoded = np.ndarray((0, w, 4))
    for f in f_list:
        print("loading file ...  " + str(i) + ' of ' + str(len(f_list)))
        this_encoded = one_hot_arr(shred_fasta(load_fasta(f), w, keep_frac))
        forw_encoded = np.append(forw_encoded ,this_encoded, axis=0)
        i+=1
    # forw_encoded = np.array(forw_encoded)
    # forw_encoded = forw_encoded.reshape(-1, 4, w)
    # rev_encoded = rev_comp_many(forw_encoded)
    return(forw_encoded)
