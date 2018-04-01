import os
import torch
import numpy as np
import pandas as pd
import pickle

all_bases = ['A', 'T', 'C', 'G']
n_bases = len(all_bases)

all_targets = ['gain', 'neutral', 'loss', 'NA']
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
    gc_num = len(np.where((mini_window[:, -2] == 'C') | (mini_window[:, -2] == 'G'))[0])
    gc_ratio = gc_num * 1.0 / len(mini_window)
    return [depth, gc_ratio]

def calculate_target_features(chrom, start, targets):
    import pdb; pdb.set_trace()
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
    features = []
    for i in range(len(window) / mini_window_size):
        start, end = i * mini_window_size, (i + 1) * mini_window_size
        feature = calculate_mini_window_feature(window[start: end])
        features.append(feature)
    return [int(window[0][1]), np.array(features)]

def load(num, input_path, target_path, window_size, mini_window_size):
    # input_ = np.loadtxt(input_path, dtype=str)
    #targets = np.loadtxt(target_path, dtype=str)
    
    if(num == None):
        input_ = pd.read_csv(input_path, delimiter="\t")
    else:
        input_ = pd.read_csv(input_path, delimiter="\t", nrows=num)

    targets = np.genfromtxt(target_path, dtype=None)
    input_ = np.asarray(input_)

    # with open(input_path) as infile:
    #     num_lines = sum(1 for line in infile)

    # chunk_size = num_lines // num_blocks
    # chunks = 1
    
    # all_windows_features = []
    # all_windows_targets = []
    
    # input_ = pd.read_csv(input_path, delimiter="\t")
    # input_ = np.asarray(input_)
    # targets = pd.read_csv(target_path, delimiter="\t")
    # targets = np.asarray(targets)
    windows = []
    tmp_window, count = [], 0
    # for input_ in pd.read_csv(input_path, delimiter="\t", chunksize=chunk_size):
        # print("Processing chunk: {}".format(chunks))
        # chunks+=1
        # input_ = np.asarray(input_)
    print("Getting windows features")
    for i in range(len(input_)):
        if i == 0 or (input_[i, 0] == input_[i-1, 0] and int(input_[i, 1]) - int(input_[i-1, 1]) == 1):
            tmp_window.append(input_[i])
            count += 1 
            if count == window_size:
                windows.append(tmp_window)
                tmp_window, count = [], 0
    
    print("Calculating mini window features")
    windows_features = []
    windows_targets = []
    for i in range(len(windows)):
        chrom = windows[i][0][0]
        features = calculate_mini_window(windows[i], mini_window_size)
        windows_features.append(features)
        target = calculate_target_features(chrom, features[0], targets)
        windows_targets.append(target)
    
    print("Calculating gaps")
    gaps = np.zeros(len(windows_features))
    for i in range(1, len(windows_features)):
        gaps[i] = int((windows_features[i][0] - windows_features[i-1][0] - window_size) / window_size)

    for i in range(len(windows_features)):
        windows_features[i][0] = gaps[i]

    # all_windows_features += windows_features
    # all_windows_targets += windows_targets
    return windows_features, windows_targets

    # return all_windows_features, all_windows_targets

    # x = np.zeros((num, seq_len, 5))
    # y = np.zeros((num, 1))
    # dist = np.zeros((num, 1))
    # dist[0] = 0
    # # x = torch.zeros((num, seq_len, 5))
    # # y = torch.zeros((num, 1))
    # cur_seq_len = 0
    # cur_num = 0
    # prev_in_pos = -1

    # with open(input_path, "rb") as infile:
    #     with open(target_path, "rb") as tarfile:
    #         cur_seq_len = 0
    #         cur_num = 0
    #         tar_num = 0
    #         for tarline in tarfile:
    #             tar_num+=1
    #             tarline = tarline.rstrip().split("\t")
    #             if(len(tarline) != 4):
    #                 continue
    #             for inline in infile:
    #                 inline = inline.rstrip().split("\t")
    #                 if(len(inline) != 4):
    #                     continue

    #                 #cnv_in_seq = 'No'
    #                 tar_chrom = tarline[0] if 'chr' in tarline[0] else 'chr' + tarline[0]
    #                 tar_start = int(tarline[1])
    #                 tar_end = int(tarline[2])
    #                 tar_cnv = tarline[3]

    #                 in_chrom = inline[0]
    #                 in_pos = int(inline[1])
    #                 if(cur_num > 0):
    #                     dist[cur_num] = in_pos - prev_in_pos
    #                 in_base = inline[2].upper()
    #                 in_read_depth = inline[3]
                     
    #                 # Check if chromosomes are equal
    #                 if(tar_chrom == in_chrom):
    #                     # if(tar_start <= in_pos and in_pos <= tar_end):
    #                     if(in_pos > tar_end):
    #                         break
    #                     # Check if in_base is a valid base (A, T, C or G)
    #                     if(in_base in all_bases): 
    #                         seq_tensor = seq_to_tensor(in_base)
    #                     else:
    #                         continue
    #                     rd_tensor = np.array([[[int(in_read_depth)]]])
    #                     x[cur_num][cur_seq_len] = np.concatenate((seq_tensor, rd_tensor),2)
                        
    #                     cur_seq_len+=1

    #                     # When cur_seq_len is equal to seq_len set y and increment cur_num
    #                     if(cur_seq_len == seq_len):
    #                         #import pdb; pdb.set_trace()
    #                         y[cur_num] = target_to_one_hot(tar_cnv)
    #                         cur_seq_len = 0
    #                         cur_num+=1
    #                         prev_in_pos = in_pos
    #                         cnv_in_seq = False
    #                         # if current number of training examples is equal to total number of training examples
    #                         # then return
    #                         if(cur_num == num):
    #                             print("Found %d training examples." %(num))
    #                             return x, y

    #                 # pos in input file has passed the end pos in the target file                      
    #                 # elif(in_pos > tar_end):
    #                 #     break
                        
    #                 # # pos in input file has not reached the start pos in the target file
    #                 # elif(in_pos < tar_start):
    #                 #     continue

    #                 # go to next inline if chromosomes are not equal
    #                 else:
    #                     continue
    # print("Did not load enough training examples")
    # return x, y, dist
    
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
   
    print("Loading all the data...")
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
    # x = np.load(in_path)
    # y = np.load(tar_path)

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
    # def __init__(self, path):
    def __init__(self, num, window_size, mini_window_size, paths):
        self.length = n_targets
        
        train_in_txt = "data/input_train.out"
        train_tar_txt = "data/target_train.out"

        val_in_txt = "data/input_val.out"
        val_tar_txt = "data/target_val.out"

        test_in_txt = "data/input_test.out"
        test_tar_txt = "data/target_test.out"

        train_in_path = paths['train_in']
        train_tar_path = paths['train_tar']

        val_in_path = paths['val_in']
        val_tar_path = paths['val_tar']

        test_in_path = paths['test_in']
        test_tar_path = paths['test_tar']

        try:
            train_x, train_y = load_data_from_file(train_in_path, train_tar_path)
        except:
            print("Could not load presaved training data")
            train_x, train_y = load_data(train_in_txt, train_tar_txt, num, window_size, mini_window_size)
            save_data(train_x, train_y, train_in_path, train_tar_path)
        
        try:
            val_x, val_y = load_data_from_file(val_in_path, val_tar_path)
        except:
            print("Could not load presaved validation data")
            val_x, val_y = load_data(val_in_txt, val_tar_txt, num, window_size, mini_window_size)
            save_data(val_x, val_y, val_in_path, val_tar_path)

        try:
            test_x, test_y = load_data_from_file(test_in_path, test_tar_path)
        except:
            print("Could not load presaved test data")
            test_x, test_y = load_data(test_in_txt, test_tar_txt, num, window_size, mini_window_size)
            save_data(test_x,test_y, test_in_path, test_tar_path)

        print("TARGET VALUES")
        unique, counts = np.unique(train_y, return_counts=True)
        print("TRAIN TAR: {}".format(dict(zip(unique, counts))))

        unique, counts = np.unique(val_y, return_counts=True)
        print("VAL TAR: {}".format(dict(zip(unique, counts))))

        unique, counts = np.unique(test_y, return_counts=True)
        print("TEST TAR: {}".format(dict(zip(unique, counts))))


        self.train_in = train_x
        self.train_tar = train_y

        # self.val_in = train_x
        # self.val_tar = train_y

        # self.test_in = train_x
        # self.test_tar = train_y
    
        self.val_in = val_x
        self.val_tar = val_y

        self.test_in = test_x
        self.test_tar = test_y


        

    
