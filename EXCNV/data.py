import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math

MAX_SIZE = 1000000

all_bases = ['A', 'T', 'C', 'G']
n_bases = len(all_bases)

all_targets = ['gain', 'neutral', 'loss']
n_targets = len(all_targets)

# Finds base index from all_bases
def base_to_index(base):
    return all_bases.index(base)

# Finds target index from all_targets
def target_to_index(target):
    return all_targets.index(target)

# Converts base to one hot encoding representation
def base_to_one_hot(base):
    '''
    Args:
        base = nucleotide
    Returns:
        numpy array of a one hot encoding representation
    '''
    base.upper()
    #tensor = np.zeros((1, n_bases))
    tensor = torch.zeros((1, n_bases))
    tensor[0][base_to_index(base)] = 1
    return tensor

def seq_to_tensor(seq):
    seq.upper()
    # tensor = np.zeros((len(seq), 1, n_bases))
    tensor = torch.zeros((len(seq), 1, n_bases))
    for i, base in enumerate(seq):
        tensor[i][0][base_to_index(base)] = 1
    
    return tensor

# Converts target to one hot encoding representation
def target_to_one_hot(target):
    '''
    Args:
        target = type of CNV
    Returns:
        torch tensor of a one hot encoding representation
    '''
    # tensor = np.array([all_targets.index(target)])
    tensor = torch.FloatTensor([all_targets.index(target)])

    return tensor


def calculate_mini_window_feature(mini_window):
    #import pdb; pdb.set_trace()
    mini_window = np.array(mini_window)
    depth = np.mean(mini_window[:, -1].astype(float))
    gc_num = len(np.where((mini_window[:, -2] == 'C') | (mini_window[:, -2] == 'G')| (mini_window[:, -2] == 'c') | (mini_window[:, -2] == 'g'))[0])
    gc_ratio = gc_num * 1.0 / len(mini_window)
    if math.isnan(depth) or math.isnan(gc_ratio):
        import pdb; pdb.set_trace()

    return [depth, gc_ratio]

def calculate_target_features(chrom, start, targets):
    #import pdb; pdb.set_trace()
    found = False
    for i in range(len(targets)):
        try:
            tar_chrom = int(targets[i][0])
        except:
            continue
        if(chrom == tar_chrom):
            if(start >= targets[i][1] and start < targets[i][2]):
                target = target_to_index(targets[i][3])
                found = True
                break
    if not found:
        target = 3

    return target


def calculate_mini_window(window, mini_window_size):
    #import pdb; pdb.set_trace()
    features = []
    for i in range(len(window) / mini_window_size):
        start, end = i * mini_window_size, (i + 1) * mini_window_size
        feature = calculate_mini_window_feature(window[start: end])
        features.append(feature)
    # return [int(window[0][1]), np.array(features)]
    return np.array(features)

def load(num, input_path, target_path, window_size, mini_window_size):
    
    if(num == -1):
        input_ = pd.read_csv(input_path, delimiter="\t",names="abcd")
    else:
        input_ = pd.read_csv(input_path, delimiter="\t", nrows=num, names="abcd")
    
    targets = np.genfromtxt(target_path, dtype=None)
    input_ = np.asarray(input_)
    
    
    small_targets = []
    for i in range(len(targets)):
        if(int(targets[i][2]) - int(targets[i][1]) < MAX_SIZE):
            small_targets.append(targets[i])
    
    targets = np.array(small_targets)
    tar_pos = 0
    s = []
    windows = []
    tmp_window = []
    count = 0
    windows_targets = []
    #import pdb; pdb.set_trace()
    print("Getting window features")
    for i in range(len(input_)):
        chrom = targets[tar_pos][0]
        end_point = targets[tar_pos][2]
        if str(input_[i,0]) == chrom and int(input_[i,1]) <= end_point:
            tmp_window.append(input_[i])
            count += 1
            if count == window_size:
                windows.append(tmp_window)
                tmp_window, count = [], 0
                if len(windows) > 100:
                    windows = []
                    continue
        else:
            tmp_window.append(input_[i])
            count += 1
            if count == window_size:
                windows.append(tmp_window)
            if len(windows) > 0 and len(windows) <= 100:
                s.append(windows)
                windows_targets.append(target_to_index(targets[tar_pos][3]))
            windows, tmp_window, count = [], [], 0
            if(tar_pos < len(targets)-1):
                tar_pos+=1
            else:
                break
   
    print("Calculating mini window features")
    s_features = []
    for k in range(len(s)):
        print("Looking at window {} out of {} total".format(k, len(s)))
        windows = s[k]
        windows_features = []
        for i in range(len(windows)):
            chrom = windows[i][0][0]
            features = calculate_mini_window(windows[i], mini_window_size)
            windows_features.append(features)

        s_features.append(windows_features)

    return s_features, windows_targets
    
def load_data(input_path, target_path, num, window_size, mini_window_size):
    '''
    Args:
        input_path: file path to input file containing read depth, base
        target_path: file path to target file containing CNV info (Yes, No)
        num: number of training/test/validation examples
        seq_len: length of sequence to input

    Returns:
        x: input matrix of size (num x batch_size x seq_len x 5)
        y: target matrix of size (num x 1)

    '''
   
    print("Processing data for {} and {}...".format(input_path, target_path))
    x, y = load(num, input_path, target_path, window_size, mini_window_size)
         
    return x, y

def save_data(input_data, target_data, in_out_path, tar_out_path):
    '''
    Args:
        input_data: input array of size (num training x batch size (optional) x seq len x 5)
        target_data: target array of size (num training x batch size (optional) x 1)
        in_out_path: path to save the input file to
        tar_out_path: path to save the output file to
    Returns:

    '''
    with open(in_out_path, 'wb') as in_out:
        pickle.dump(input_data, in_out)

    with open(tar_out_path, 'wb') as tar_out:
        pickle.dump(target_data, tar_out)

def load_data_from_file(in_path, tar_path):
    '''
    Args:
        in_path: path to input file
        tar_path: path to target file
    
    Returns:
        x: numpy array of input data
        y: numpy array of target data
    '''
    print("Loading data from file...")
    with open(in_path, 'r') as infile:
        x = pickle.load(infile)
    
    with open(tar_path, 'r') as tarfile:
        y = pickle.load(tarfile)

    return x, y

def rearrange(input_data):
    '''
    Args: 
        input_data: input data in the form of batch size x seq_len x input size
    
    Returns:
        new_input_data: input data in the form of seq_len x batch_size x input size
    '''

    new_input_data = torch.zeros((input_data.shape[1], input_data.shape[0], input_data.shape[2]))

    for i in range(0, input_data.shape[1]): # iterate through seq
        for j in range(0, input_data.shape[0]): # iterate through batches
            new_input_data[i][j] = input_data[j][i]
    
    return new_input_data

class Corpus(object):
    def __init__(self, num, window_size, mini_window_size, paths, data_folder):
        self.length = n_targets

        data_in_path = paths['data_in']
        data_tar_path = paths['data_tar']
        try:
            data_x, data_y = load_data_from_file(data_in_path, data_tar_path)
        except:
            print("Could not load training data")
    
        print("TARGET VALUES")
        unique, counts = np.unique(data_y, return_counts=True)
        print("DATASET TAR: {}".format(dict(zip(unique, counts))))

        self.data_x = data_x
        self.data_y = data_y
